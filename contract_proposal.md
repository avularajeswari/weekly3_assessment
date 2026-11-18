
# TECHNICAL PROPOSAL
## AI-Based Contract Question Answering System

### 1. Problem Statement

Organizations often need to search lengthy contracts to find
important information such as payment terms, termination
conditions, penalties, renewal dates and obligations.

Reading a 60-page contract manually takes time and makes
information retrieval difficult.

The proposed solution is an AI-powered Contract Question
Answering system using Retrieval-Augmented Generation (RAG).

### 2. Recommended Technical Approach

Use a Retrieval-Augmented Generation (RAG) architecture.

System workflow:

1. Upload the 60-page PDF contract.
2. Extract text using a PDF parsing library.
3. Divide the extracted text into smaller chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in a vector database.
6. Accept questions in natural language.
7. Retrieve the most relevant contract sections.
8. Send the retrieved sections to an LLM.
9. Generate an answer with page references and citations.
10. If sufficient evidence is unavailable, return
    "Information not found in the contract."

Recommended technologies:

- Python: Backend development
- PyMuPDF: PDF text extraction
- Embedding model: Text embeddings
- FAISS: Vector search
- LLM: Answer generation
- Streamlit: User interface

### 3. Data Requirements

- Original 60-page PDF contract.
- Extracted contract text.
- Page numbers and section headings.
- Chunked contract passages.
- Embeddings for semantic retrieval.
- A set of 50-100 verified question-answer pairs
  prepared with legal expert assistance.

Sensitive contract information must be protected using
access controls and approved data storage.

### 4. Evaluation Metrics and Targets

The system will be evaluated using a verified test dataset.

| Metric | Target |
| Retrieval Recall@5 | At least 90% |
| Answer correctness | At least 90% |
| Citation accuracy | At least 95% |
| Unsupported answer rate | Below 5% |

These are proposed acceptance targets, not measured results.
The system must be tested before deployment.

### 5. Estimated Build Time

| Development activity | Time |
| PDF extraction and preprocessing | 3 days |
| Chunking and embedding | 3 days |
| Vector database and retrieval | 4 days |
| LLM integration | 3 days |
| User interface | 3 days |
| Testing and evaluation | 4 days |
| Total | Approximately 20 working days |

### 6. Key Trade-offs

Advantages:
- Faster contract information retrieval.
- Answers supported by contract passages.
- Reduced manual searching.
- Easier access to lengthy documents.

Limitations:
- Incorrect answers are still possible.
- Poor PDF extraction can reduce accuracy.
- Embedding and LLM calls have costs.
- Legal interpretation still requires an expert.

### 7. Human Review and Safety

Legal professionals must review high-impact answers,
especially termination, penalties, liability and payment clauses.

The system must not provide final legal advice,
approve agreements or make legal decisions automatically.

### 8. Conclusion

A RAG-based contract Q&A system provides an efficient
way to retrieve information from lengthy contracts.

Combining document retrieval, an LLM, page-level citations
and human legal review makes the solution more useful,
traceable and reliable than relying on an LLM alone.
