import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_bge_encoder():
    try:
        from FlagEmbedding import BGEM3FlagModel
        logger.info("Loading BAAI/bge-m3 via FlagEmbedding...")
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

        class BGEWrapper:
            def __init__(self, m):
                self.model = m

            def encode(self, texts, batch_size=16, show_progress_bar=False):
                if isinstance(texts, str):
                    texts = [texts]
                out = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    emb = self.model.encode(
                        batch, batch_size=batch_size, max_length=512,
                        return_dense=True, return_sparse=False, return_colbert_vecs=False
                    )['dense_vecs']
                    out.append(emb)
                return np.vstack(out)

        return BGEWrapper(model)

    except ImportError:
        logger.warning("FlagEmbedding not installed. Using sentence-transformers.")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('BAAI/bge-m3')

class BM25BGERRFRetriever:

    RRF_K = 60 

    def __init__(self,
                 gnn_data_dir: str = "gnn_data",
                 processed_dir: str = "dataset_processed",
                 rules_dir: str = "official_documents"):
        self.gnn_data_dir = Path(gnn_data_dir)
        self.processed_dir = Path(processed_dir)
        self.rules_dir = Path(rules_dir)

        logger.info("Initializing BM25 + BGE-M3 RRF Retriever...")
        self.cases = []
        self.case_id_to_idx = {}

        self._load_cases()
        self._load_official_pdfs()
        self._build_bm25_index()
        self._load_bge_encoder()
        self._build_bge_embeddings()
        logger.info(f"Ready: {len(self.cases)} total documents indexed")

    def _load_cases(self):
        cache_path = self.gnn_data_dir / 'cases_metadata_cache.json'
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                self.cases = cache['cases']
                self.case_id_to_idx = cache['case_id_to_idx']
                logger.info(f"Loaded {len(self.cases)} cases from cache")
                return
            except Exception as e:
                logger.warning(f"Cache failed: {e}")

        courts = ['supreme_court', 'delhi_high_court', 'bombay_high_court',
                  'calcutta_high_court', 'allahabad_high_court', 'madras_high_court']
        for court in courts:
            court_dir = self.processed_dir / court
            if not court_dir.exists():
                continue
            for json_file in tqdm(list(court_dir.glob('*.json')),
                                  desc=f"Loading {court}", leave=False):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    idx = len(self.cases)
                    case_id = json_file.stem
                    self.cases.append({
                        'id': case_id, 'file_name': data['file_name'],
                        'court': court, 'metadata': data['metadata'],
                        'text': data['text'], 'doc_type': 'case',
                        'word_count': data.get('word_count', 0)
                    })
                    self.case_id_to_idx[case_id] = idx
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
        logger.info(f"Loaded {len(self.cases)} cases")

    def _load_official_pdfs(self):
        cache_path = self.gnn_data_dir / 'pdf_chunks_cache.json'
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    for chunk in json.load(f):
                        self.cases.append(chunk)
                return
            except Exception as e:
                logger.warning(f"PDF cache failed: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer with lowercasing."""
        return text.lower().split()

    def _build_bm25_index(self):
        logger.info("Building BM25 index...")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 not installed. Run: pip install rank_bm25")

        tokenized = [
            self._tokenize(c['text'][:2048] if c.get('text') else c['metadata'].get('title', ''))
            for c in self.cases
        ]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(tokenized)} documents")

    def _load_bge_encoder(self):
        logger.info("Loading BGE M3-Embedding encoder...")
        self.bge_encoder = load_bge_encoder()
        logger.info("BGE M3-Embedding loaded")

    def _build_bge_embeddings(self):
        cache_path = self.gnn_data_dir / 'text_embeddings_bge_m3.npy'
        if cache_path.exists():
            logger.info("Loading cached BGE M3 embeddings...")
            self.bge_embeddings = np.load(cache_path)
            logger.info(f"Cached: {self.bge_embeddings.shape}")
            return
        logger.info("Building BGE M3 embeddings (first run — may take a few minutes)...")
        texts = [
            (c['text'][:512] if c.get('text') else c['metadata'].get('title', ''))
            for c in self.cases
        ]
        batch_size = 16
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with BGE M3"):
            batch_embs = self.bge_encoder.encode(texts[i:i + batch_size],
                                                  batch_size=batch_size,
                                                  show_progress_bar=False)
            all_embs.append(batch_embs)
        self.bge_embeddings = np.vstack(all_embs)
        np.save(cache_path, self.bge_embeddings)
        logger.info(f"BGE M3 embeddings built and cached: {self.bge_embeddings.shape}")

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank + 1)

    def _reciprocal_rank_fusion(self,
                                bm25_ranking: List[int],
                                bge_ranking: List[int]) -> Dict[int, float]:
      
        rrf_scores: Dict[int, float] = {}
        for rank, doc_idx in enumerate(bm25_ranking):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + self._rrf_score(rank)
        for rank, doc_idx in enumerate(bge_ranking):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + self._rrf_score(rank)
        return rrf_scores

    def retrieve(self, query: str, top_k: int = 5,
                 candidate_pool: int = 1000) -> List[Dict]:
       
        logger.info(f"[BM25+BGE RRF] Query: '{query[:60]}...'")

        logger.info("  BM25 ranking...")
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = np.argsort(bm25_scores)[-candidate_pool:][::-1].tolist()

        logger.info("  BGE M3 dense ranking...")
        from sklearn.metrics.pairwise import cosine_similarity
        q_emb = self.bge_encoder.encode([query], batch_size=1, show_progress_bar=False)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        bge_scores = cosine_similarity(q_emb, self.bge_embeddings)[0]
        bge_top = np.argsort(bge_scores)[-candidate_pool:][::-1].tolist()

        logger.info("  RRF fusion...")
        rrf_scores = self._reciprocal_rank_fusion(bm25_top, bge_top)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (doc_idx, rrf_score) in enumerate(ranked):
            case = self.cases[doc_idx]
            results.append({
                'rank': rank + 1,
                'case_id': case['id'],
                'file_name': case['file_name'],
                'court': case['court'],
                'title': case['metadata'].get('title', 'Unknown'),
                'date': case['metadata'].get('date', 'Unknown'),
                'text_snippet': case['text'][:500] if case.get('text') else '',
                'score': rrf_score,
                'bm25_score': float(bm25_scores[doc_idx]),
                'bge_score': float(bge_scores[doc_idx]),
                'doc_type': case.get('doc_type', 'case'),
                'word_count': case.get('word_count', 0),
                'retriever': 'bm25_bge_rrf'
            })

        logger.info(f"  Returned {len(results)} results")
        return results

    def retrieve_bm25_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25-only retrieval for ablation comparison."""
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_indices):
            case = self.cases[idx]
            results.append({
                'rank': rank + 1, 'case_id': case['id'],
                'title': case['metadata'].get('title', 'Unknown'),
                'court': case['court'],
                'date': case['metadata'].get('date', 'Unknown'),
                'text_snippet': case['text'][:500] if case.get('text') else '',
                'score': float(bm25_scores[idx]),
                'doc_type': case.get('doc_type', 'case'),
                'retriever': 'bm25_only'
            })
        return results

    def retrieve_bge_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """BGE M3 dense-only retrieval for ablation comparison."""
        from sklearn.metrics.pairwise import cosine_similarity
        q_emb = self.bge_encoder.encode([query], batch_size=1, show_progress_bar=False)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        bge_scores = cosine_similarity(q_emb, self.bge_embeddings)[0]
        top_indices = np.argsort(bge_scores)[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_indices):
            case = self.cases[idx]
            results.append({
                'rank': rank + 1, 'case_id': case['id'],
                'title': case['metadata'].get('title', 'Unknown'),
                'court': case['court'],
                'date': case['metadata'].get('date', 'Unknown'),
                'text_snippet': case['text'][:500] if case.get('text') else '',
                'score': float(bge_scores[idx]),
                'doc_type': case.get('doc_type', 'case'),
                'retriever': 'bge_m3_only'
            })
        return results

if __name__ == "__main__":
    retriever = BM25BGERRFRetriever(
        gnn_data_dir="gnn_data",
        processed_dir="dataset_processed",
        rules_dir="official_documents"
    )

    test_queries = [
        "What punishment is there for murder under IPC?",
        "Bail conditions for non-bailable offences under CrPC",
        "Property rights of women after divorce in India",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        print("\n[BM25 Only]")
        for r in retriever.retrieve_bm25_only(query, top_k=3):
            print(f"  {r['rank']}. {r['title'][:60]} | Score: {r['score']:.4f}")

        print("\n[BGE M3 Only]")
        for r in retriever.retrieve_bge_only(query, top_k=3):
            print(f"  {r['rank']}. {r['title'][:60]} | Score: {r['score']:.4f}")

        print("\n[BM25 + BGE M3 RRF]")
        for r in retriever.retrieve(query, top_k=3):
            print(f"  {r['rank']}. {r['title'][:60]} | RRF: {r['score']:.4f}")
