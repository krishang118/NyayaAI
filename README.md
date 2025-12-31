# Nyaya AI: A Neuro-Symbolic AI Legal System

An AI-driven legal research system built for Indian jurisprudence, combining structured legal knowledge with neural representation learning to support accurate case retrieval, statutory analysis, and explainable legal reasoning across Indian courts.

## Overview

Nyaya AI (Sanskrit: न्याय, meaning "Justice") is an advanced legal intelligence system designed specifically for Indian jurisprudence. It combines symbolic legal knowledge with modern neural architectures to provide intelligent case retrieval, statutory analysis, and transparent legal reasoning.

### Key Innovations

- Large-Scale Legal Corpus: 56,025 cases from Supreme Court + 5 High Courts (2000-2024)
- Integrated Knowledge Graph: 154,068 nodes & 725,563 edges combining case law with statutory provisions
- Graph Neural Networks: GAT model achieving 96.55% F1-score on legal entity classification
- Two-Stage Neurosymbolic Retrieval: Hybrid approach combining text (70%), GAT context (15%), and symbolic legal features (15%)
- Reasoning Transparency: Local and secure DeepSeek-R1 integration providing explainable legal analysis without internet
- Production-Ready Interface: Beautiful Streamlit-based chatbot with easy-to-use UI

###

<p align="center">
  <img src="Demo1.gif" width="240" />
  <img src="Demo2.gif" width="240" />
  <img src="Demo3.gif" width="240" />
</p>

## Features

### Core Capabilities

- Semantic Legal Search: SBERT-powered semantic similarity across 56K+ cases and statutory provisions
- Citation Network Analysis: PageRank-based importance scoring and precedent discovery
- Court Hierarchy Awareness: Automatic prioritization of Supreme Court over High Court judgments
- Statutory Grounding: Integration with IPC, CrPC, Constitution, and Evidence Act provisions
- Multi-hop Reasoning: Graph traversal for discovering indirect legal connections
- Transparent Explanations: Step-by-step reasoning process with citation provenance

### Coverage

- Courts: Supreme Court, Delhi HC, Bombay HC, Calcutta HC, Allahabad HC, Madras HC
- Timespan: 2000-2024 (25 years)
- Statutory Laws: Indian Penal Code, Code of Criminal Procedure, Constitution of India, Indian Evidence Act
- Legal Domains: Criminal law, Constitutional law, Civil law, Tax law, Service matters

## System Architecture

```
Web Scraping from Indian Kanoon (56k Cases) → Preprocessing to obtain JSON files → Knowledge Graph Construction based on IndiLegalOnt for the case and statutory law files (154k nodes, 725k edges) → GNN Training based on KG (3 layers, 4 attention heads) → Two-Stage Retrieval [Text Search (70) + Neurosymbolic Reranking (GAT+Symbolic, 15+15)] → Local DeepSeek-R1 LLM Reasoning → Chat Interface and Visualizations via Streamlit UI
```

### Pipeline Stages

1. Data Collection: Automated web scraping with Selenium + BeautifulSoup
2. Knowledge Graph Construction: Entity extraction + ontology-based structuring
3. Graph Neural Network Training: GAT-based representation learning
4. Two-Stage Retrieval: Hybrid neurosymbolic ranking
5. LLM Integration: Response generation with reasoning transparency

## The Dataset

### Statistics

| Metric | Value |
|--------|-------|
| Total Cases | 56,025 |
| Supreme Court | 9,979 |
| High Courts | 46,046 |
| Years Covered | 2000-2024 |
| Knowledge Graph Nodes | 154,068 |
| Knowledge Graph Edges | 725,563 |
| Statutory Provisions | 1,684 |

### Court Distribution

```
Supreme Court of India    : 9,979 cases
Delhi High Court          : 8,821 cases
Bombay High Court         : 9,436 cases
Calcutta High Court       : 9,584 cases
Allahabad High Court      : 8,398 cases
Madras High Court         : 9,807 cases
```

### Data Sources

- Case Law: [Indian Kanoon](https://indiankanoon.org)
- Statutory Laws: Official Government of India PDFs
  - Indian Penal Code (IPC)
  - Code of Criminal Procedure (CrPC)
  - Constitution of India
  - Indian Evidence Act

## Results

### Retrieval Performance

| Metric | Score |
|--------|-------|
| Precision@5 | 0.89 |
| NDCG@5 | 0.93 |
| Response Quality | 4.61/5.0 |

### GAT Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 94.48% |
| Test F1-Score | 96.55% |
| Precision | 99.22% |
| Recall | 94.48% |

### Comparative Analysis

| Approach | P@5 | NDCG@5 |
|----------|-----|--------|
| Text-only (SBERT) | 0.74 | 0.82 |
| Text + Symbolic | 0.83 | 0.88 |
| Nyaya AI (Full) | 0.89 | 0.93 |

## Example Queries

```
Q: What punishment is there for murder under IPC?
→ Retrieves Section 302 IPC + Supreme Court precedents
```

```
Q: How to legally evict a tenant?
→ Provides step-by-step procedure with case law
```

```
Q: Search for defamation cases
→ Returns relevant cases with citation network analysis
```

## How to Run

1. Make sure Python 3.8+ is installed.
2. Clone this repository on your local machine.
3. Install the required dependencies:
```bash
pip install streamlit==1.28.0 sentence-transformers==2.2.2 transformers==4.30.0 networkx==3.1 matplotlib==3.7.1 pandas==2.0.3 numpy==1.24.3 scipy==1.11.1 beautifulsoup4==4.12.2 lxml==4.9.3 selenium==4.15.2 webdriver-manager==4.0.1 requests==2.31.0 tqdm==4.65.0 PyPDF2==3.0.1 spacy==3.6.1 owlready2==0.43 scikit-learn==1.3.0 ollama==0.1.7
```
4. Install Ollama & set up DeepSeek-R1 (7b).
5. Run all the files, one-by-one, starting from data extraction, then building the KG, and training the GNN. After that, you can directly run the chatbot python file to access the system via the streamlit interface.

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License. 
