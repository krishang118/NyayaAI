import numpy as np
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import ollama
from tqdm import tqdm
import pickle
import os
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_bge_encoder():
    try:
        from FlagEmbedding import BGEM3FlagModel
        logger.info("Loading BAAI/bge-m3 via FlagEmbedding...")
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

        class BGEWrapper:
            def __init__(self, model):
                self.model = model
                self.embedding_dim = 1024

            def encode(self, texts, batch_size=16, show_progress_bar=False):
                if isinstance(texts, str):
                    texts = [texts]
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    output = self.model.encode(
                        batch,
                        batch_size=batch_size,
                        max_length=512,
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False
                    )
                    all_embeddings.append(output['dense_vecs'])
                return np.vstack(all_embeddings)

        return BGEWrapper(model)

    except ImportError:
        logger.warning("FlagEmbedding not installed. Falling back to sentence-transformers.")
        from sentence_transformers import SentenceTransformer
        logger.info("Loading BAAI/bge-m3 via sentence-transformers...")
        model = SentenceTransformer('BAAI/bge-m3')
        return model

class NeurosymbolicLegalRetriever:

    def __init__(self,
                 gnn_data_dir: str = "gnn_data",
                 processed_dir: str = "dataset_processed",
                 rules_dir: str = "official_documents"):
        self.gnn_data_dir = Path(gnn_data_dir)
        self.processed_dir = Path(processed_dir)
        self.rules_dir = Path(rules_dir)
        self.provisions_loaded = False

        logger.info("Initializing Two-Stage Neurosymbolic Legal Retriever (BGE M3)...")
        self.cases = []
        self.case_id_to_idx = {}

        self.load_knowledge_graph()
        self.load_gat_embeddings()
        self.load_case_metadata()
        self.load_official_pdfs()
        self.initialize_text_encoder()
        self.build_text_embeddings()
        self.build_case_to_node_mapping()

        logger.info(
            f"Ready: {sum(1 for c in self.cases if c.get('doc_type') == 'case')} cases, "
            f"{sum(1 for c in self.cases if c.get('doc_type') == 'provision')} provisions, "
            f"{sum(1 for c in self.cases if c.get('doc_type') == 'pdf_document')} PDF chunks"
        )

    def load_knowledge_graph(self):
        logger.info("Loading knowledge graph...")
        kg_path = self.gnn_data_dir / 'knowledge_graph.gpickle'
        if not kg_path.exists():
            raise FileNotFoundError(f"Knowledge graph not found: {kg_path}")
        with open(kg_path, 'rb') as f:
            self.G = pickle.load(f)
        logger.info(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def load_gat_embeddings(self):
        logger.info("Loading GAT embeddings...")
        emb_path = self.gnn_data_dir / 'final_embeddings_node_type.npy'
        if not emb_path.exists():
            logger.warning("GAT embeddings not found — GAT scores will be zero.")
            self.gat_embeddings = None
            self.node_metadata = []
            self.node_id_to_gat_idx = {}
            return
        self.gat_embeddings = np.load(emb_path)
        with open(self.gnn_data_dir / 'node_metadata.json', 'r') as f:
            self.node_metadata = json.load(f)
        self.node_id_to_gat_idx = {
            meta['id']: idx for idx, meta in enumerate(self.node_metadata)
        }
        logger.info(f"GAT embeddings: {self.gat_embeddings.shape}")

    def load_case_metadata(self):
        logger.info("Loading case metadata...")
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
                logger.warning(f"Cache load failed: {e}")

        self.cases = []
        self.case_id_to_idx = {}
        courts = ['supreme_court', 'delhi_high_court', 'bombay_high_court',
                  'calcutta_high_court', 'allahabad_high_court', 'madras_high_court']
        for court in courts:
            court_dir = self.processed_dir / court
            if not court_dir.exists():
                continue
            for json_file in tqdm(list(court_dir.glob('*.json')), desc=f"Loading {court}", leave=False):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    idx = len(self.cases)
                    case_id = json_file.stem
                    self.cases.append({
                        'id': case_id, 'file_name': data['file_name'],
                        'court': court, 'metadata': data['metadata'],
                        'text': data['text'], 'text_length': data.get('text_length', 0),
                        'word_count': data.get('word_count', 0), 'doc_type': 'case'
                    })
                    self.case_id_to_idx[case_id] = idx
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
        logger.info(f"Loaded {len(self.cases)} cases")
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'cases': self.cases, 'case_id_to_idx': self.case_id_to_idx}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to cache: {e}")

    def load_official_pdfs(self):
        logger.info("Loading official legal documents...")
        cache_path = self.gnn_data_dir / 'pdf_chunks_cache.json'
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    for chunk in json.load(f):
                        self.cases.append(chunk)
                return
            except Exception as e:
                logger.warning(f"PDF cache load failed: {e}")
        pdf_files = {
            'ipc': 'Indian Penal Code.pdf', 'crpc': 'Code of Criminal Procedure.pdf',
            'constitution': 'Constitution of India.pdf', 'evidence': 'Indian Evidence Act.pdf'
        }
        try:
            import PyPDF2
        except ImportError:
            logger.warning("PyPDF2 not installed — skipping PDF loading.")
            return
        pdf_chunks_to_cache = []
        for doc_type, filename in pdf_files.items():
            filepath = self.rules_dir / filename
            if not filepath.exists():
                continue
            try:
                with open(filepath, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    full_text = "".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                chunks = self._chunk_document(full_text, doc_type)
                for i, chunk in enumerate(chunks):
                    entry = {
                        'id': f"{doc_type}_chunk_{i}", 'file_name': f"{filename}_chunk_{i}",
                        'court': 'statutory',
                        'metadata': {
                            'title': f"{filename.replace('.pdf','')} - Part {i+1}",
                            'date': 'Statutory', 'citations': [],
                            'source_pdf': filename, 'chunk_number': i + 1
                        },
                        'text': chunk, 'text_length': len(chunk),
                        'word_count': len(chunk.split()), 'doc_type': 'pdf_document'
                    }
                    self.cases.append(entry)
                    pdf_chunks_to_cache.append(entry)
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")
        if pdf_chunks_to_cache:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(pdf_chunks_to_cache, f, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to cache PDFs: {e}")

    def _chunk_document(self, text: str, doc_type: str, chunk_size: int = 1000) -> List[str]:
        chunks = []
        if doc_type == 'constitution':
            parts = text.split('Article')
            chunks = [f"Article{p[:chunk_size]}" for p in parts[1:] if len(p) > 50]
        elif doc_type in ['ipc', 'crpc', 'evidence']:
            parts = text.split('Section')
            chunks = [f"Section{p[:chunk_size]}" for p in parts[1:] if len(p) > 50]
        if not chunks or len(chunks) < 5:
            chunks = []
            words = text.split()
            current, length = [], 0
            for word in words:
                current.append(word)
                length += len(word) + 1
                if length >= chunk_size:
                    chunks.append(' '.join(current))
                    current, length = [], 0
            if current:
                chunks.append(' '.join(current))
        if not chunks:
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size) if text[i:i + chunk_size].strip()]
        return chunks

    def initialize_text_encoder(self):
        logger.info("Initializing text encoder: BAAI/bge-m3 (1024-dim)...")
        try:
            self.text_encoder = load_bge_encoder()
            logger.info("BGE M3-Embedding loaded successfully (1024-dim)")
        except Exception as e:
            logger.error(f"Failed to load BGE M3: {e}")
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.warning("Fell back to MiniLM-L6-v2")

    def build_text_embeddings(self):
      
        logger.info("Building BGE M3 text embeddings...")
        cache_path = self.gnn_data_dir / 'text_embeddings_bge_m3.npy'
        if cache_path.exists():
            logger.info("Loading cached BGE M3 embeddings...")
            self.text_embeddings = np.load(cache_path)
            logger.info(f"Cached embeddings: {self.text_embeddings.shape}")
            return
        if self.text_encoder is None:
            self.text_embeddings = None
            return
        texts = [
            (c['text'][:512] if c['text'] else c['metadata'].get('title', ''))
            for c in self.cases
        ]
        batch_size = 16 
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with BGE M3"):
            batch = texts[i:i + batch_size]
            batch_emb = self.text_encoder.encode(batch, batch_size=batch_size, show_progress_bar=False)
            embeddings.append(batch_emb)
        self.text_embeddings = np.vstack(embeddings)
        np.save(cache_path, self.text_embeddings)
        logger.info(f"BGE M3 embeddings built: {self.text_embeddings.shape}")

    def build_case_to_node_mapping(self):
        logger.info("Building case-to-node mapping...")
        cache_path = self.gnn_data_dir / 'case_to_node_mapping.json'
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    self.case_idx_to_node_id = {int(k): v for k, v in json.load(f).items()}
                return
            except Exception as e:
                logger.warning(f"Mapping cache load failed: {e}")
        self.case_idx_to_node_id = {}
        for idx, case in enumerate(self.cases):
            node_id_1 = f"Case_{idx+1}"
            if node_id_1 in self.G.nodes:
                self.case_idx_to_node_id[idx] = node_id_1
            else:
                for node in self.G.nodes():
                    if self.G.nodes[node].get('case_id') == case['id']:
                        self.case_idx_to_node_id[idx] = node
                        break
        logger.info(f"Mapped {len(self.case_idx_to_node_id)} documents to graph nodes")
        try:
            with open(cache_path, 'w') as f:
                json.dump({str(k): v for k, v in self.case_idx_to_node_id.items()}, f)
        except Exception as e:
            logger.warning(f"Failed to cache mapping: {e}")

    def get_symbolic_scores(self, case_indices: List[int]) -> np.ndarray:
        scores = np.zeros(len(case_indices))
        for i, case_idx in enumerate(case_indices):
            case = self.cases[case_idx]
            if case.get('doc_type') == 'pdf_document':
                scores[i] = 0.3
                continue
            node_id = self.case_idx_to_node_id.get(case_idx)
            if node_id and node_id in self.G.nodes:
                nd = self.G.nodes[node_id]
                pr = min(nd.get('pagerank', 0) * 100, 1.0)
                court = case['court']
                cs = 1.0 if 'supreme_court' in court else (0.6 if 'high_court' in court else 0.3)
                cit = min(nd.get('cited_by_count', 0) / 10, 1.0)
                try:
                    rec = max(0, min(1, (int(nd.get('year', 2000)) - 2000) / 25))
                except Exception:
                    rec = 0.3
                scores[i] = 0.25 * pr + 0.35 * cs + 0.25 * cit + 0.15 * rec
            else:
                court = case['court']
                scores[i] = 0.8 if 'supreme_court' in court else (0.5 if 'high_court' in court else 0.3)
        return scores

    def get_gat_context_scores(self, case_indices: List[int]) -> np.ndarray:
        if self.gat_embeddings is None:
            return np.zeros(len(case_indices))
        scores = np.zeros(len(case_indices))
        context_embeddings = []
        for case_idx in case_indices:
            node_id = self.case_idx_to_node_id.get(case_idx)
            if node_id and node_id in self.G.nodes and node_id in self.node_id_to_gat_idx:
                gat_idx = self.node_id_to_gat_idx[node_id]
                case_emb = self.gat_embeddings[gat_idx]
                try:
                    neighbors = list(self.G.successors(node_id)) + list(self.G.predecessors(node_id))
                    neighbor_embs = [case_emb] + [
                        self.gat_embeddings[self.node_id_to_gat_idx[n]]
                        for n in neighbors if n in self.node_id_to_gat_idx
                    ]
                    weights = [2.0] + [1.0] * (len(neighbor_embs) - 1)
                    context_emb = np.average(neighbor_embs, axis=0, weights=weights)
                except Exception:
                    context_emb = case_emb
            else:
                context_emb = np.zeros(self.gat_embeddings.shape[1])
            context_embeddings.append(context_emb)
        context_embeddings = np.array(context_embeddings)
        valid = context_embeddings[context_embeddings.sum(axis=1) != 0]
        if len(valid) > 0:
            centroid = np.mean(valid, axis=0)
            for i, emb in enumerate(context_embeddings):
                if emb.sum() != 0:
                    scores[i] = cosine_similarity([emb], [centroid])[0][0]
        return scores

    def retrieve(self, query: str, top_k: int = 5, stage1_k: int = 100,
                 alpha_text: float = 0.70,
                 alpha_gat: float = 0.15,
                 alpha_symbolic: float = 0.15) -> List[Dict]:
    
        logger.info(f"[BGE M3] Two-stage retrieval: '{query[:60]}...'")

        # Stage 1: dense retrieval with BGE M3
        query_emb = self.text_encoder.encode([query], batch_size=1, show_progress_bar=False)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        text_scores = cosine_similarity(query_emb, self.text_embeddings)[0]

        stage1_indices = np.argsort(text_scores)[-stage1_k:][::-1]
        stage1_text_scores = text_scores[stage1_indices]

        # Stage 2: GAT + symbolic reranking
        gat_scores = self.get_gat_context_scores(stage1_indices.tolist())
        sym_scores = self.get_symbolic_scores(stage1_indices.tolist())

        def normalize(s):
            lo, hi = s.min(), s.max()
            return (s - lo) / (hi - lo + 1e-8)

        hybrid_scores = (
            alpha_text * normalize(stage1_text_scores) +
            alpha_gat * normalize(gat_scores) +
            alpha_symbolic * normalize(sym_scores)
        )
        top_idx = np.argsort(hybrid_scores)[-top_k:][::-1]
        final_indices = stage1_indices[top_idx]

        results = []
        for rank, (idx, hi) in enumerate(zip(final_indices, top_idx)):
            case = self.cases[idx]
            node_id = self.case_idx_to_node_id.get(idx)
            neighbors_info = ""
            if node_id and node_id in self.G.nodes:
                try:
                    neighbors_info = (
                        f"Cited by {self.G.in_degree(node_id)} cases, "
                        f"cites {self.G.out_degree(node_id)} cases"
                    )
                except Exception:
                    pass
            results.append({
                'rank': rank + 1, 'case_id': case['id'],
                'file_name': case['file_name'], 'court': case['court'],
                'title': case['metadata'].get('title', 'Unknown'),
                'date': case['metadata'].get('date', 'Unknown'),
                'text_snippet': case['text'][:500] if case['text'] else '',
                'score': float(hybrid_scores[hi]),
                'text_score': float(normalize(stage1_text_scores)[hi]),
                'gat_score': float(normalize(gat_scores)[hi]),
                'symbolic_score': float(normalize(sym_scores)[hi]),
                'node_id': node_id, 'neighbors_info': neighbors_info,
                'citations': case['metadata'].get('citations', [])[:5],
                'word_count': case.get('word_count', 0),
                'doc_type': case.get('doc_type', 'case'),
                'encoder': 'bge-m3'
            })
        return results

class LegalChatbot:
    def __init__(self, retriever: NeurosymbolicLegalRetriever,
                 llm_model: str = "deepseek-r1:7b"):
        self.retriever = retriever
        self.llm_model = llm_model
        logger.info(f"LegalChatbot (BGE M3) with LLM: {llm_model}")
        try:
            ollama.list()
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            raise

    def format_context(self, retrieved_cases: List[Dict]) -> str:
        context = ""
        for i, case in enumerate(retrieved_cases, 1):
            doc_type = case.get('doc_type', 'case')
            if doc_type == 'provision':
                context += (
                    f"\nStatutory Provision {i}: {case['title']}\n"
                    f"Source: {case['metadata'].get('parent_act', 'Unknown Act')}\n"
                    f"Relevance: {case['score']:.3f} "
                    f"(Text: {case['text_score']:.3f}, GAT: {case['gat_score']:.3f}, "
                    f"Symbolic: {case['symbolic_score']:.3f})\n"
                    f"\nProvision Text:\n{case['text_snippet'][:600]}...\n"
                )
            elif doc_type == 'pdf_document':
                context += (
                    f"\nStatutory Text {i}: {case['title']}\n"
                    f"Source: {case['metadata'].get('source_pdf', 'Unknown')}\n"
                    f"Relevance: {case['score']:.3f}\n"
                    f"\nExcerpt:\n{case['text_snippet'][:700]}...\n"
                )
            else:
                context += (
                    f"\nCase {i}: {case['title']}\n"
                    f"Court: {case['court'].replace('_',' ').title()} | Date: {case['date']}\n"
                    f"Relevance: {case['score']:.3f} "
                    f"(Text: {case['text_score']:.3f}, GAT: {case['gat_score']:.3f}, "
                    f"Symbolic: {case['symbolic_score']:.3f})\n"
                )
                if case['neighbors_info']:
                    context += f"Citation Network: {case['neighbors_info']}\n"
                context += f"\nSummary:\n{case['text_snippet'][:400]}...\n"
        return context

    def chat(self, query: str, top_k: int = 5, stage1_k: int = 100,
             alpha_text: float = 0.70, alpha_gat: float = 0.15,
             alpha_symbolic: float = 0.15, return_thinking: bool = True) -> Dict:
        retrieved = self.retriever.retrieve(
            query, top_k=top_k, stage1_k=stage1_k,
            alpha_text=alpha_text, alpha_gat=alpha_gat, alpha_symbolic=alpha_symbolic
        )
        context = self.format_context(retrieved)
        prompt = f"""You are an expert Indian legal research assistant.

USER QUERY:
{query}

RETRIEVED LEGAL CONTEXT (BGE M3 + GAT + Symbolic):
Stage 1: BGE M3-Embedding semantic retrieval ({stage1_k} candidates, 1024-dim)
Stage 2: Re-ranked using Text ({alpha_text*100:.0f}%), GAT ({alpha_gat*100:.0f}%), Symbolic ({alpha_symbolic*100:.0f}%)

{context}

INSTRUCTIONS:
1. Provide a direct answer grounded in the retrieved documents.
2. Cite specific case names and statutory provisions.
3. Distinguish binding Supreme Court precedent from persuasive High Court authority.
4. Flag jurisdictional variations where applicable.
5. Conclude with appropriate caveats about consulting a qualified legal professional.
"""
        try:
            resp = ollama.generate(model=self.llm_model, prompt=prompt,
                                   options={'temperature': 0.7, 'num_ctx': 8192})
            response_text = resp['response']
            thinking = None
            if return_thinking and '<thinking>' in response_text:
                import re
                m = re.search(r'<thinking>(.*?)</thinking>', response_text, re.DOTALL)
                if m:
                    thinking = m.group(1).strip()
                    response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text,
                                           flags=re.DOTALL).strip()
            return {
                'query': query, 'response': response_text, 'thinking': thinking,
                'retrieved_cases': retrieved, 'context': context,
                'retrieval_config': {
                    'encoder': 'bge-m3', 'stage1_candidates': stage1_k,
                    'final_top_k': top_k,
                    'weights': {'text': alpha_text, 'gat': alpha_gat, 'symbolic': alpha_symbolic}
                }
            }
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {'query': query, 'response': f"Error: {e}", 'thinking': None,
                    'retrieved_cases': retrieved, 'context': context}

    def interactive_chat(self):
        print("\nNyaya AI (BGE M3 variant)")
        print("Two-Stage: BGE M3 (70%) + GAT Context (15%) + Symbolic (15%)")
        print("Type 'quit' to exit\n")
        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye.")
                    break
                result = self.chat(query)
                print(f"\nAssistant:\n{result['response']}")
                if result['thinking']:
                    preview = result['thinking'][:500]
                    print(f"\nReasoning:\n{preview}{'...' if len(result['thinking']) > 500 else ''}")
                print(f"\nTop 3 results:")
                for case in result['retrieved_cases'][:3]:
                    print(f"  {case['rank']}. {case['title'][:70]}")
                    print(f"     Score: {case['score']:.3f} | {case['court'].replace('_',' ').title()}")
                print()
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    try:
        retriever = NeurosymbolicLegalRetriever(
            gnn_data_dir="gnn_data",
            processed_dir="dataset_processed",
            rules_dir="official_documents"
        )
        chatbot = LegalChatbot(retriever, llm_model="deepseek-r1:7b")
        query = "What punishment is there for murder under IPC?"
        print(f"\nQuery: {query}\n")
        result = chatbot.chat(query, top_k=5, stage1_k=100)
        print(f"Response:\n{result['response']}")
        print(f"\nTop 5 results (BGE M3):")
        for case in result['retrieved_cases']:
            print(f"  {case['rank']}. {case['title'][:70]}")
            print(f"     Score: {case['score']:.3f} | Encoder: {case['encoder']}")

        if input("\nStart interactive chat? (y/n): ").strip().lower() in ['y', 'yes']:
            chatbot.interactive_chat()

    except FileNotFoundError as e:
        logger.error(f"Setup incomplete: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
