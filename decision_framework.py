
import csv

# ============================================
# DECISION FRAMEWORK AND PROPOSAL
# ============================================

problems = [
    {
        "Problem": "GST calculation on invoices",
        "Input Type": "Invoice amount, GST rate, CGST/SGST or IGST",
        "Output Type": "Tax amount and final invoice total",
        "Latency": "Very low",
        "Cost": "Very low",
        "Explainability": "Very high",
        "Tool Choice": "Python rules engine and calculator",
        "Human Review": "Review unusual tax cases and exceptions",
        "AI Should NOT": "Do not use an LLM for tax arithmetic or final tax decisions"
    },
    {
        "Problem": "Employee leave balance",
        "Input Type": "Employee ID, leave records, approved leave policy",
        "Output Type": "Available leave days and leave summary",
        "Latency": "Low",
        "Cost": "Low",
        "Explainability": "High",
        "Tool Choice": "HRMS database, SQL and rule-based calculation",
        "Human Review": "HR reviews disputes and policy exceptions",
        "AI Should NOT": "Do not let an LLM invent balances or approve leave"
    },
    {
        "Problem": "Ranking resumes for a role",
        "Input Type": "Candidate resumes and job description",
        "Output Type": "Ranked candidate list with matching skills",
        "Latency": "Medium",
        "Cost": "Medium",
        "Explainability": "Very high",
        "Tool Choice": "NLP/LLM for skill extraction and transparent ranking",
        "Human Review": "Recruiter reviews every shortlist and hiring decision",
        "AI Should NOT": "Do not automatically reject candidates or use protected attributes"
    },
    {
        "Problem": "Answering questions from a 60-page contract",
        "Input Type": "PDF contract and natural-language questions",
        "Output Type": "Answers with supporting clauses and page references",
        "Latency": "Medium",
        "Cost": "Medium",
        "Explainability": "Very high",
        "Tool Choice": "RAG, vector database and LLM",
        "Human Review": "Legal professional reviews important answers",
        "AI Should NOT": "Do not invent clauses or make final legal decisions"
    },
    {
        "Problem": "Forecasting next month's sales",
        "Input Type": "Historical sales, dates, seasonality and promotions",
        "Output Type": "Next-month sales forecast and uncertainty range",
        "Latency": "Medium",
        "Cost": "Medium",
        "Explainability": "High",
        "Tool Choice": "Time-series forecasting (ARIMA/Prophet) and validation",
        "Human Review": "Sales manager reviews forecast and business assumptions",
        "AI Should NOT": "Do not make major financial commitments automatically"
    }
]


# ============================================
# 1. CREATE DECISION TABLE CSV
# ============================================

with open(
    "decision_framework.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=problems[0].keys()
    )

    writer.writeheader()
    writer.writerows(problems)

print("Decision framework CSV created successfully!")


# ============================================
# 2. CREATE ONE-PAGE TECHNICAL PROPOSAL
# ============================================

proposal = """
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
"""

with open(
    "contract_proposal.md",
    "w",
    encoding="utf-8"
) as file:

    file.write(proposal)

print("Technical proposal created successfully!")


# ============================================
# 3. CREATE 3-MINUTE PRESENTATION SCRIPT
# ============================================

presentation = """
3-MINUTE PRESENTATION SCRIPT

Good morning everyone.

Today I am presenting my decision framework and
technical proposal for five business problems.

First, GST calculation on invoices.

For this problem, I recommend Python-based rules and
calculations instead of an LLM because tax calculations
require accuracy and predictable results.

Second, employee leave balance.

I recommend an HRMS database with SQL and business rules.
The official leave balance should come from company records,
and HR should review exceptions.

Third, resume ranking.

NLP and language models can extract skills and compare
candidate qualifications with job requirements.
However, recruiters must make the final hiring decisions.
AI should not automatically reject candidates.

Fourth, contract question answering.

I recommend Retrieval-Augmented Generation, also called RAG.
It retrieves relevant contract passages and uses an LLM
to generate answers with supporting page references.
Legal professionals must review important answers.

Fifth, sales forecasting.

I recommend time-series forecasting using historical sales
data. A sales manager should review the forecast before
making important business decisions.

Now I will explain my technical proposal for contract Q&A.

The proposed system accepts a 60-page PDF contract.
It extracts text, splits the document into chunks,
creates embeddings and stores them in a vector database.

When a user asks a question, the system retrieves
the most relevant passages and sends them to an LLM.
The LLM generates an answer with supporting citations.

My proposed evaluation targets are:
at least 90 percent retrieval recall at five,
90 percent answer correctness,
95 percent citation accuracy,
and less than 5 percent unsupported answers.

The estimated development time is approximately
20 working days.

The main trade-offs are API cost, document extraction
quality, retrieval accuracy and the possibility of
incorrect AI-generated answers.

Finally, a legal expert must review important answers.

Thank you.
"""

with open(
    "presentation_script.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(presentation)

print("Presentation script created successfully!")


# ============================================
# 4. DISPLAY RESULTS
# ============================================

print("\n========== DECISION FRAMEWORK ==========\n")

for index, problem in enumerate(problems, start=1):

    print(f"{index}. {problem['Problem']}")
    print("Input:", problem["Input Type"])
    print("Output:", problem["Output Type"])
    print("Tool:", problem["Tool Choice"])
    print("Human Review:", problem["Human Review"])
    print("AI Restriction:", problem["AI Should NOT"])
    print("-" * 55)


print("\n========== PROJECT COMPLETED ==========")
print("Files generated:")
print("1. decision_framework.csv")
print("2. contract_proposal.md")
print("3. presentation_script.txt")
