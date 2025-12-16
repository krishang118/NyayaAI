import numpy as np
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer
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
class NeurosymbolicLegalRetriever:
    def __init__(self, gnn_data_dir: str = "gnn_data", 
                 processed_dir: str = "dataset_processed",
                 rules_dir: str = "official_documents"):
        self.gnn_data_dir = Path(gnn_data_dir)
        self.processed_dir = Path(processed_dir)
        self.rules_dir = Path(rules_dir)
        logger.info("Initializing Two-Stage Neurosymbolic Legal Retriever...")
        self.cases = []
        self.case_id_to_idx = {}        
        self.load_knowledge_graph()
        self.load_gat_embeddings()
        self.load_case_metadata() 
        self.load_official_pdfs()  
        self.initialize_text_encoder()
        self.build_text_embeddings()
        self.build_case_to_node_mapping()
        case_count = sum(1 for c in self.cases if c.get('doc_type', 'case') == 'case')
        provision_count = sum(1 for c in self.cases if c.get('doc_type') == 'provision')
        pdf_count = sum(1 for c in self.cases if c.get('doc_type') == 'pdf_document')    
    def load_knowledge_graph(self):
        logger.info("Loading knowledge graph...")
        kg_path = self.gnn_data_dir / 'knowledge_graph.gpickle'
        if not kg_path.exists():
            logger.error(f"Knowledge graph not found at {kg_path}")
            raise FileNotFoundError(f"Missing: {kg_path}")
        with open(kg_path, 'rb') as f:
            self.G = pickle.load(f)
        logger.info(f"Loaded graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
    def load_gat_embeddings(self):
        logger.info("Loading GAT embeddings...")        
        emb_path = self.gnn_data_dir / 'final_embeddings_node_type.npy'
        if not emb_path.exists():
            logger.warning(f"GAT embeddings not found at {emb_path}")
            self.gat_embeddings = None
            self.node_metadata = []
            self.node_id_to_gat_idx = {}
            return
        self.gat_embeddings = np.load(emb_path)
        with open(self.gnn_data_dir / 'node_metadata.json', 'r') as f:
            self.node_metadata = json.load(f)        
        self.node_id_to_gat_idx = {}
        for idx, node_meta in enumerate(self.node_metadata):
            node_id = node_meta['id']
            self.node_id_to_gat_idx[node_id] = idx
        logger.info(f"Loaded GAT embeddings: {self.gat_embeddings.shape}")
    def load_case_metadata(self):
        logger.info("Loading case metadata...")    
        cache_path = self.gnn_data_dir / 'cases_metadata_cache.json'    
        if cache_path.exists():
            logger.info("Loading cached case metadata...")
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                self.cases = cache_data['cases']
                self.case_id_to_idx = cache_data['case_id_to_idx']
                initial_case_count = len(self.cases)
                logger.info(f"Loaded {initial_case_count} cases from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached case metadata: {e}")
        self.cases = []
        self.case_id_to_idx = {}
        courts = ['supreme_court', 'delhi_high_court', 'bombay_high_court',
                  'calcutta_high_court', 'allahabad_high_court', 'madras_high_court']
        for court in courts:
            court_dir = self.processed_dir / court
            if not court_dir.exists():
                continue
            json_files = list(court_dir.glob('*.json'))
            for json_file in tqdm(json_files, desc=f"Loading {court}", leave=False):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)
                    case_id = json_file.stem
                    idx = len(self.cases)
                    self.cases.append({
                        'id': case_id, 'file_name': case_data['file_name'],
                        'court': court, 'metadata': case_data['metadata'],
                        'text': case_data['text'], 'text_length': case_data.get('text_length', 0),
                        'word_count': case_data.get('word_count', 0), 'doc_type': 'case'})
                    self.case_id_to_idx[case_id] = idx
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
        initial_case_count = len(self.cases)
        logger.info(f"Loaded {initial_case_count} cases")
        try:
            cache_data = {
                'cases': self.cases, 'case_id_to_idx': self.case_id_to_idx}
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False)
            logger.info(f"Cached case metadata to: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache metadata: {e}")
    def load_official_pdfs(self):
        logger.info("Loading official legal documents...")
        cache_path = self.gnn_data_dir / 'pdf_chunks_cache.json'
        if cache_path.exists():
            logger.info("Loading cached PDF chunks...")
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    pdf_chunks = json.load(f)
                total_chunks = 0
                for chunk in pdf_chunks:
                    self.cases.append(chunk)
                    total_chunks += 1
                return
            except Exception as e:
                logger.warning(f"Failed to load cached PDF chunks: {e}")
        pdf_files = {
            'ipc': 'Indian Penal Code.pdf', 'crpc': 'Code of Criminal Procedure.pdf',
            'constitution': 'Constitution of India.pdf', 'evidence': 'Indian Evidence Act.pdf'}
        try:
            import PyPDF2
        except ImportError:
            logger.error("PyPDF2 not installed.")
            logger.warning("Skipping PDF loading. Official documents won't be searchable.")
            return
        pdf_chunks_to_cache = []    
        total_chunks = 0
        for doc_type, filename in pdf_files.items():
            filepath = self.rules_dir / filename
            if not filepath.exists():
                logger.warning(f"PDF not found: {filepath}")
                continue
            try:
                logger.info(f"  Reading {filename}...")
                with open(filepath, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    full_text = ""                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
                    if not full_text.strip():
                        logger.warning(f"  No text extracted from {filename}")
                        continue                    
                    chunks = self._chunk_document(full_text, doc_type, chunk_size=1000)
                    for i, chunk in enumerate(chunks):
                        chunk_data = {
                            'id': f"{doc_type}_chunk_{i}", 'file_name': f"{filename}_chunk_{i}",
                            'court': 'statutory',
                            'metadata': {
                                'title': f"{filename.replace('.pdf', '')} - Part {i+1}",
                                'date': 'Statutory','citations': [],
                                'source_pdf': filename,'chunk_number': i+1},
                            'text': chunk,'text_length': len(chunk),
                            'word_count': len(chunk.split()),
                            'doc_type': 'pdf_document'}
                        self.cases.append(chunk_data)
                        pdf_chunks_to_cache.append(chunk_data)         
                        total_chunks += 1
            except Exception as e:
                logger.error(f"  Error reading {filename}: {e}")
                continue
        if pdf_chunks_to_cache:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(pdf_chunks_to_cache, f, ensure_ascii=False)
                logger.info(f"Cached {total_chunks} PDF chunks to: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache PDF chunks: {e}")
    def _chunk_document(self, text: str, doc_type: str, chunk_size: int = 1000) -> List[str]:
        chunks = []        
        if doc_type == 'constitution':
            parts = text.split('Article')
            for i, part in enumerate(parts[1:], 1):
                if len(part) > 50: 
                    chunk_text = f"Article{part[:chunk_size]}"
                    chunks.append(chunk_text)
        elif doc_type in ['ipc', 'crpc', 'evidence']:
            parts = text.split('Section')
            for i, part in enumerate(parts[1:], 1):
                if len(part) > 50: 
                    chunk_text = f"Section{part[:chunk_size]}"
                    chunks.append(chunk_text)        
        if not chunks or len(chunks) < 5:
            chunks = []
            words = text.split()
            current_chunk = []
            current_length = 0
            for word in words:
                current_chunk.append(word)
                current_length += len(word) + 1
                if current_length >= chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        if not chunks:
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
        return chunks
    def load_official_provisions(self):
        if self.provisions_loaded:
            logger.debug("Provisions already loaded, skipping...")
            return
        logger.info("Loading provisions from official documents...")        
        initial_count = len(self.cases)
        provision_index_path = self.gnn_data_dir / 'provision_index.json'
        if not provision_index_path.exists():
            logger.warning("Provision index not found. Trying to load from KG...")
            self.extract_provisions_from_graph()
            self.provisions_loaded = True
            return        
        try:
            with open(provision_index_path, 'r') as f:
                provision_index = json.load(f)
            provisions_added = 0
            for doc_type, provisions in provision_index.items():
                for prov_num, prov_data in provisions.items():
                    idx = len(self.cases)                    
                    provision_text = f"{prov_data['title']}\n\n{prov_data['text']}"
                    self.cases.append({
                        'id': f"{doc_type}_{prov_num}", 'file_name': f"{doc_type}_{prov_num}",
                        'court': 'statutory',
                        'metadata': {
                            'title': f"{prov_data['act']} - Section/Article {prov_num}",
                            'date': 'Statutory', 'citations': [],
                            'provision_number': prov_num,
                            'parent_act': prov_data['act']},
                        'text': provision_text,
                        'text_length': len(provision_text),
                        'word_count': len(provision_text.split()),
                        'doc_type': 'provision'})
                    provisions_added += 1
            self.provisions_loaded = True            
        except Exception as e:
            logger.warning(f"Could not load provision index: {e}")
            self.extract_provisions_from_graph()
            self.provisions_loaded = True
    def extract_provisions_from_graph(self):
        if self.provisions_loaded:
            logger.debug("Provisions already loaded, skipping extraction...")
            return
        logger.info("Extracting provisions from knowledge graph...")
        initial_count = len(self.cases)
        provisions_added = 0
        for node_id, node_data in self.G.nodes(data=True):
            if node_data.get('node_type') == 'provision':
                label = node_data.get('label', '')
                parent_act = node_data.get('parent_act', 'Unknown Act')
                rulebook = node_data.get('rulebook', '')
                provision_text = f"{parent_act} - {label}\n\n[Provision text from {rulebook}]"
                self.cases.append({
                    'id': f"provision_{node_id}",
                    'file_name': node_id,'court': 'statutory',
                    'metadata': {
                        'title': f"{parent_act} - {label}",
                        'date': 'Statutory', 'citations': [],
                        'parent_act': parent_act},
                    'text': provision_text, 'text_length': len(provision_text),
                    'word_count': len(provision_text.split()),
                    'doc_type': 'provision'})
                provisions_added += 1
        logger.info(f"Extracted {provisions_added} provisions from graph (total docs: {len(self.cases)})")
        self.provisions_loaded = True        
        self.load_official_provisions()
    def initialize_text_encoder(self):
        logger.info("Initializing text encoder...")
        try:
            self.text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Loaded: all-MiniLM-L6-v2 (384 dim)")
        except:
            logger.warning("Could not load text encoder")
            self.text_encoder = None
    def build_text_embeddings(self):
        logger.info("Building text embeddings...")
        cache_path = self.gnn_data_dir / 'text_embeddings.npy'
        if cache_path.exists():
            logger.info("Loading cached text embeddings...")
            self.text_embeddings = np.load(cache_path)
            logger.info(f"Loaded cached embeddings: {self.text_embeddings.shape}")
            return
        if self.text_encoder is None:
            logger.warning("No text encoder available, skipping text embeddings")
            self.text_embeddings = None
            return
        texts = []
        for case in self.cases:
            text_snippet = case['text'][:512] if case['text'] else case['metadata'].get('title', '')
            texts.append(text_snippet)        
        batch_size = 32
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i:i+batch_size]
            batch_emb = self.text_encoder.encode(batch, show_progress_bar=False)
            embeddings.append(batch_emb)
        self.text_embeddings = np.vstack(embeddings)
        np.save(cache_path, self.text_embeddings)
        logger.info(f"Built text embeddings: {self.text_embeddings.shape}")
        logger.info(f"Cached to: {cache_path}")
    def build_case_to_node_mapping(self):
        logger.info("Building case-to-node mapping...")
        cache_path = self.gnn_data_dir / 'case_to_node_mapping.json'
        if cache_path.exists():
            logger.info("Loading cached case-to-node mapping...")
            try:
                with open(cache_path, 'r') as f:
                    cached_mapping = json.load(f)
                    self.case_idx_to_node_id = {int(k): v for k, v in cached_mapping.items()}
                return
            except Exception as e:
                logger.warning(f"Failed to load cached mapping: {e}")
        self.case_idx_to_node_id = {}
        for idx, case in enumerate(self.cases):
            if case.get('doc_type') == 'provision':
                parent_act = case['metadata'].get('parent_act', '')
                prov_num = case['metadata'].get('provision_number', '')                
                for node in self.G.nodes():
                    node_data = self.G.nodes[node]
                    if node_data.get('node_type') == 'provision':
                        node_label = node_data.get('label', '')
                        node_act = node_data.get('parent_act', '')
                        if prov_num and str(prov_num) in node_label and parent_act in node_act:
                            self.case_idx_to_node_id[idx] = node
                            break
                continue
            node_id_1 = f"Case_{idx+1}"
            court = case['court']
            file_name = case['file_name']
            if node_id_1 in self.G.nodes:
                self.case_idx_to_node_id[idx] = node_id_1
            else:
                for node in self.G.nodes():
                    node_data = self.G.nodes[node]
                    if node_data.get('case_id') == case['id']:
                        self.case_idx_to_node_id[idx] = node
                        break        
        case_count = sum(1 for c in self.cases if c.get('doc_type') == 'case')
        provision_count = sum(1 for c in self.cases if c.get('doc_type') == 'provision')
        mapped_count = len(self.case_idx_to_node_id)
        logger.info(f"Mapped {mapped_count} documents to graph nodes")  
        try:
            with open(cache_path, 'w') as f:
                json.dump({str(k): v for k, v in self.case_idx_to_node_id.items()}, f)
            logger.info(f"Cached mapping to: {cache_path}")
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
                node_data = self.G.nodes[node_id]                
                pagerank = node_data.get('pagerank', 0)
                pagerank_normalized = min(pagerank * 100, 1.0)
                court = case['court']
                if 'supreme_court' in court.lower():
                    court_score = 1.0
                elif 'high_court' in court.lower():
                    court_score = 0.6
                else:
                    court_score = 0.3
                cited_by = node_data.get('cited_by_count', 0)
                citation_score = min(cited_by / 10, 1.0)                
                year = node_data.get('year', '2000')
                try:
                    year_int = int(year)
                    recency_score = max(0, min(1, (year_int - 2000) / 25))
                except:
                    recency_score = 0.3                 
                symbolic_score = (
                    pagerank_normalized * 0.25 + court_score * 0.35 +
                    citation_score * 0.25 + recency_score * 0.15)
            else:
                court = case['court']
                if 'supreme_court' in court.lower():
                    symbolic_score = 0.8  
                elif 'high_court' in court.lower():
                    symbolic_score = 0.5 
                else:
                    symbolic_score = 0.3 
            scores[i] = symbolic_score
        return scores
    def get_gat_context_scores(self, case_indices: List[int]) -> np.ndarray:
        if self.gat_embeddings is None:
            return np.zeros(len(case_indices))
        scores = np.zeros(len(case_indices))        
        context_embeddings = []
        valid_indices = []
        for case_idx in case_indices:
            node_id = self.case_idx_to_node_id.get(case_idx)
            if node_id and node_id in self.G.nodes and node_id in self.node_id_to_gat_idx:
                gat_idx = self.node_id_to_gat_idx[node_id]
                case_emb = self.gat_embeddings[gat_idx]                
                try:
                    cited_nodes = list(self.G.successors(node_id))
                    citing_nodes = list(self.G.predecessors(node_id))                    
                    neighbor_embs = [case_emb]                    
                    for neighbor in cited_nodes + citing_nodes:
                        if neighbor in self.node_id_to_gat_idx:
                            neighbor_gat_idx = self.node_id_to_gat_idx[neighbor]
                            neighbor_embs.append(self.gat_embeddings[neighbor_gat_idx])                    
                    if len(neighbor_embs) > 1:
                        weights = [2.0] + [1.0] * (len(neighbor_embs) - 1)
                        context_emb = np.average(neighbor_embs, axis=0, weights=weights)
                    else:
                        context_emb = case_emb
                    context_embeddings.append(context_emb)
                    valid_indices.append(len(context_embeddings) - 1)
                except:
                    context_embeddings.append(case_emb)
                    valid_indices.append(len(context_embeddings) - 1)
            else:
                context_embeddings.append(np.zeros(self.gat_embeddings.shape[1]))
                valid_indices.append(len(context_embeddings) - 1)
        context_embeddings = np.array(context_embeddings)        
        if len(context_embeddings) > 1:
            valid_contexts = context_embeddings[context_embeddings.sum(axis=1) != 0]
            if len(valid_contexts) > 0:
                centroid = np.mean(valid_contexts, axis=0)                
                for i, context_emb in enumerate(context_embeddings):
                    if context_emb.sum() != 0:
                        scores[i] = cosine_similarity([context_emb], [centroid])[0][0]
        return scores
    def retrieve(self, query: str, top_k: int = 10, 
                stage1_k: int = 100,
                alpha_text: float = 0.7, 
                alpha_gat: float = 0.15, 
                alpha_symbolic: float = 0.15) -> List[Dict]:
        logger.info(f"Two-stage retrieval for query: '{query[:50]}...'")        
        logger.info(f"  Stage 1: Text retrieval...")
        if self.text_embeddings is not None and self.text_encoder is not None:
            query_emb = self.text_encoder.encode([query])[0]
            text_scores = cosine_similarity([query_emb], self.text_embeddings)[0]
        else:
            logger.error("Text embeddings not available")
            return []
        stage1_indices = np.argsort(text_scores)[-stage1_k:][::-1]
        stage1_text_scores = text_scores[stage1_indices]
        logger.info(f"  Stage 2: GAT + Symbolic re-ranking...")        
        gat_scores = self.get_gat_context_scores(stage1_indices.tolist())
        symbolic_scores = self.get_symbolic_scores(stage1_indices.tolist())
        def normalize(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s < 1e-8:
                return np.ones_like(scores) * 0.5
            return (scores - min_s) / (max_s - min_s)
        text_scores_norm = normalize(stage1_text_scores)
        gat_scores_norm = normalize(gat_scores)
        symbolic_scores_norm = normalize(symbolic_scores)        
        hybrid_scores = (
            alpha_text * text_scores_norm + alpha_gat * gat_scores_norm + 
            alpha_symbolic * symbolic_scores_norm)        
        top_reranked_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        final_indices = stage1_indices[top_reranked_indices]
        results = []
        for rank, idx in enumerate(final_indices):
            case = self.cases[idx]
            reranked_idx = top_reranked_indices[rank]
            node_id = self.case_idx_to_node_id.get(idx, None)            
            neighbors_info = ""
            if node_id and node_id in self.G.nodes:
                try:
                    citing_count = self.G.in_degree(node_id)
                    cited_count = self.G.out_degree(node_id)
                    neighbors_info = f"Cited by {citing_count} cases, cites {cited_count} cases"
                except:
                    pass
            results.append({
                'rank': rank + 1, 'case_id': case['id'],
                'file_name': case['file_name'],'court': case['court'],
                'title': case['metadata'].get('title', 'Unknown'),
                'date': case['metadata'].get('date', 'Unknown'),
                'text_snippet': case['text'][:500] if case['text'] else '',
                'score': float(hybrid_scores[reranked_idx]),
                'text_score': float(text_scores_norm[reranked_idx]),
                'gat_score': float(gat_scores_norm[reranked_idx]),
                'symbolic_score': float(symbolic_scores_norm[reranked_idx]),
                'node_id': node_id,'neighbors_info': neighbors_info,
                'citations': case['metadata'].get('citations', [])[:5],
                'word_count': case['word_count']})
        avg_text = np.mean([r['text_score'] for r in results])
        avg_gat = np.mean([r['gat_score'] for r in results])
        avg_symbolic = np.mean([r['symbolic_score'] for r in results])
        return results
class LegalChatbot:    
    def __init__(self, retriever: NeurosymbolicLegalRetriever, 
                 llm_model: str = "deepseek-r1:7b"):
        self.retriever = retriever
        self.llm_model = llm_model
        logger.info(f"Initializing Legal Chatbot with LLM: {llm_model}")
        try:
            models = ollama.list()
            logger.info(f"Ollama connected.")
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            raise    
    def format_context(self, retrieved_cases: List[Dict]) -> str:
        context = ""        
        for i, case in enumerate(retrieved_cases, 1):
            doc_type = case.get('doc_type', 'case')
            if doc_type == 'provision':
                context += f"\n"
                context += f"Statuary Provision {i}: {case['title']}\n"
                context += f"\n"
                context += f"Source: {case['metadata'].get('parent_act', 'Unknown Act')}\n"
                context += f"Type: Statutory Law\n"
                context += f"Relevance Score: {case['score']:.3f}\n"
                context += f"  └─ Text Similarity: {case['text_score']:.3f}\n"
                context += f"  └─ Graph Context (GAT): {case['gat_score']:.3f}\n"
                context += f"  └─ Citation Importance: {case['symbolic_score']:.3f}\n"
                if case['neighbors_info']:
                    context += f"Usage in Cases: {case['neighbors_info']}\n"
                context += f"\nProvision Text:\n{case['text_snippet'][:600]}...\n"
            elif doc_type == 'pdf_document':
                context += f"\n"
                context += f"Statuary Text {i}: {case['title']}\n"
                context += f"\n"
                context += f"Source: {case['metadata'].get('source_pdf', 'Unknown')}\n"
                context += f"Type: Official Legal Document\n"
                context += f"Relevance Score: {case['score']:.3f}\n"
                context += f"  └─ Text Similarity: {case['text_score']:.3f}\n"
                context += f"\nExcerpt:\n{case['text_snippet'][:700]}...\n"
            else:
                context += f"\n"
                context += f"Case {i}: {case['title']}\n"
                context += f"\n"
                context += f"Court: {case['court'].replace('_', ' ').title()}\n"
                context += f"Date: {case['date']}\n"
                context += f"Overall Relevance: {case['score']:.3f}\n"
                context += f"  └─ Text Similarity: {case['text_score']:.3f}\n"
                context += f"  └─ Graph Context (GAT): {case['gat_score']:.3f}\n"
                context += f"  └─ Citation Importance: {case['symbolic_score']:.3f}\n"
                if case['neighbors_info']:
                    context += f"Citation Network: {case['neighbors_info']}\n"
                if case['citations']:
                    context += f"\nKey Citations:\n"
                    for cite in case['citations'][:3]:
                        context += f"  • {cite[:80]}\n"
                context += f"\nSummary:\n{case['text_snippet'][:400]}...\n"
        return context
    def chat(self, query: str, top_k: int = 5, 
            stage1_k: int = 100, alpha_text: float = 0.7,
            alpha_gat: float = 0.15, alpha_symbolic: float = 0.15,
            return_thinking: bool = True) -> Dict:
        logger.info(f"\n")
        logger.info(f"Processing Query...")
        logger.info(f"Query: {query}\n")        
        logger.info("Step 1/3: Two-stage neurosymbolic retrieval...")
        retrieved_cases = self.retriever.retrieve(
            query, top_k=top_k,
            stage1_k=stage1_k, alpha_text=alpha_text,
            alpha_gat=alpha_gat, alpha_symbolic=alpha_symbolic)        
        logger.info("Step 2/3: Building context...")
        context = self.format_context(retrieved_cases)        
        logger.info(f"Step 3/3: Generating response with {self.llm_model}...")
        prompt = f"""You are an expert Indian legal research assistant with deep knowledge of case law, statutes, and legal precedents.

USER QUERY:
{query}

RETRIEVED CASES (Two-Stage Neurosymbolic Retrieval):
These cases were retrieved using a sophisticated two-stage approach:

STAGE 1 (Text Search): Semantic similarity identified {stage1_k} candidates using SBERT embeddings
  → Searches case law, statutory provisions, AND official legal documents (IPC, CrPC, Constitution, Evidence Act)
STAGE 2 (Re-ranking): The top {top_k} results were selected by combining:
  • Text Similarity ({alpha_text*100:.0f}%): Semantic relevance to your query
  • Graph Context ({alpha_gat*100:.0f}%): GAT embeddings capturing citation network structure
  • Citation Importance ({alpha_symbolic*100:.0f}%): PageRank, court hierarchy, and citation counts

NOTE: Results may include case law (judicial decisions), statutory provisions (from graph), and direct excerpts from official legal documents (IPC, CrPC, Constitution, Evidence Act PDFs).

{context}

INSTRUCTIONS:
Provide a comprehensive legal research response:

1. **Direct Answer**: Address the query based on the retrieved cases

2. **Case Analysis**: 
   - Identify the most relevant case(s) and explain why
   - Note how the GAT graph context scores influenced ranking
   - Highlight key legal principles and precedents
   - Discuss any conflicting interpretations

3. **Citation Network Context**:
   - Consider how cases cite each other (shown in "Citation Network" info)
   - Explain precedent relationships
   - Note which cases are more influential (higher citation counts)

4. **Court Hierarchy**:
   - Supreme Court decisions are binding on all lower courts
   - High Court decisions are persuasive authority
   - Explain which court's decision carries more weight

5. **Recommendation**:
   - Provide actionable guidance based on case law
   - Note any caveats or limitations

**Be specific, cite case names, and explain your legal reasoning clearly.**
"""
        try:
            response = ollama.generate(
                model=self.llm_model, prompt=prompt,
                options={
                    'temperature': 0.7, 'num_ctx': 8192})
            response_text = response['response']            
            thinking = None
            if return_thinking and '<thinking>' in response_text:
                import re
                thinking_match = re.search(r'<thinking>(.*?)</thinking>', 
                                          response_text, re.DOTALL)
                if thinking_match:
                    thinking = thinking_match.group(1).strip()
                    response_text = re.sub(r'<thinking>.*?</thinking>', '', 
                                          response_text, flags=re.DOTALL).strip()            
            logger.info("Response generated successfully.")
            return {
                'query': query,'response': response_text,
                'thinking': thinking, 'retrieved_cases': retrieved_cases,
                'context': context,
                'retrieval_config': {
                    'stage1_candidates': stage1_k,
                    'final_top_k': top_k,
                    'weights': {
                        'text': alpha_text,
                        'gat': alpha_gat,'symbolic': alpha_symbolic}}}
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                'query': query,
                'response': f"Error generating response: {e}",
                'thinking': None, 'retrieved_cases': retrieved_cases,
                'context': context}
    def interactive_chat(self):
        print("\n")
        print("Indian Legal Case Recommender Chatbot :")
        print("Two-Stage Neurosymbolic Retrieval + DeepSeek-R1 -")
        print("\nRetrieval: Text (70%) + GAT Context (15%) + Symbolic (15%) :-")
        print("Enter your legal query...")
        print("Type 'quit' to exit, 'help' for commands\n")
        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye.")
                    break
                if query.lower() == 'help':
                    print("\nCommands:")
                    print("  - Type your legal query naturally")
                    print("  - 'quit' or 'exit' - Exit the chatbot")
                    print("  - 'help' - Show this message")
                    print()
                    continue
                result = self.chat(
                    query, top_k=5,
                    stage1_k=100, alpha_text=0.7,
                    alpha_gat=0.15, alpha_symbolic=0.15)                
                print("\n")
                print("Assistant:\n")
                print(result['response'])
                if result['thinking']:
                    print("\n")
                    print("Reasoning Process:\n")
                    print(result['thinking'][:500] + "..." if len(result['thinking']) > 500 else result['thinking'])
                print("\n")
                print(f"Retrieved Cases (Two-Stage Retrieval):")
                print(f"    Stage 1: {result['retrieval_config']['stage1_candidates']} text candidates")
                print(f"    Stage 2: Re-ranked to top {result['retrieval_config']['final_top_k']}")
                print(f"\n    Top 3 Results:")
                for i, case in enumerate(result['retrieved_cases'][:3], 1):
                    print(f"\n  {i}. {case['title'][:70]}")
                    print(f"     Court: {case['court'].replace('_', ' ').title()} | Date: {case['date']}")
                    print(f"     Overall: {case['score']:.3f} (Text: {case['text_score']:.3f}, GAT: {case['gat_score']:.3f}, Symbolic: {case['symbolic_score']:.3f})")
                    if case['neighbors_info']:
                        print(f"     {case['neighbors_info']}")
                print("\n")
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
if __name__ == "__main__":    
    try:
        retriever = NeurosymbolicLegalRetriever(
            gnn_data_dir="gnn_data", processed_dir="dataset_processed",
            rules_dir="official_documents")
        chatbot = LegalChatbot(
            retriever=retriever, llm_model="deepseek-r1:7b")
        print("\n")
        print("Example Query - Demonstrating Two-Stage Retrieval")
        print("\n")
        example_query = "What punishment is there for murder under IPC?"
        print(f"Query: {example_query}\n")
        result = chatbot.chat(
            example_query, top_k=5,
            stage1_k=100, alpha_text=0.7,
            alpha_gat=0.15, alpha_symbolic=0.15)
        print("\n")
        print("ChatBot Response:")
        print("\n")
        print(result['response'])
        if result['thinking']:
            print("\n")
            print("Reasoning Process:")
            print("\n")
            thinking_preview = result['thinking'][:600]
            print(thinking_preview + "..." if len(result['thinking']) > 600 else thinking_preview)
        print("\n")
        print("Retrieval Analysis:")
        print("\n")
        print(f"Stage 1: Retrieved {result['retrieval_config']['stage1_candidates']} candidates via text search")
        print(f"Stage 2: Re-ranked to top {result['retrieval_config']['final_top_k']} using GAT + Symbolic")       
        print(f"\nTop 5 Retrieved Cases:")
        for case in result['retrieved_cases']:
            print(f"\n{case['rank']}. {case['title'][:70]}")
            print(f"   Court: {case['court'].replace('_', ' ').title()} | Date: {case['date']}")
            if case['neighbors_info']:
                print(f"   Citation Network: {case['neighbors_info']}")
        print("\n")
        print("Example completed.")
        response = input("Start interactive chat mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("\n")
            print("Starting Interactive Mode...")
            chatbot.interactive_chat()
        else:
            print("\nTo start interactive mode later, run:")
            print("  chatbot.interactive_chat()\n")        
    except FileNotFoundError as e:
        logger.error(f"\nSetup incomplete: {e}")
    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
