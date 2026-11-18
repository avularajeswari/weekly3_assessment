import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# STEP 1: CREATE 12 FAQ QUESTIONS
# ============================================================

faqs = [

    # HR FAQs
    {
        "category": "HR",
        "question": "What are the work from home policies?",
        "answer": "Employees can work from home according to company policy. Approval from the manager is required."
    },

    {
        "category": "HR",
        "question": "How many annual leaves are employees allowed?",
        "answer": "Employees receive annual leave according to company leave policy."
    },

    {
        "category": "HR",
        "question": "How can I apply for sick leave?",
        "answer": "Employees can apply for sick leave through the HR portal."
    },

    {
        "category": "HR",
        "question": "How do I contact the human resources department?",
        "answer": "You can contact HR through the official company email or employee portal."
    },

    # IT FAQs
    {
        "category": "IT",
        "question": "How can I reset my account password?",
        "answer": "Use the forgot password option on the company login page to reset your password."
    },

    {
        "category": "IT",
        "question": "What should I do if I cannot log in to my account?",
        "answer": "Check your login credentials. If the issue continues, contact the IT support team."
    },

    {
        "category": "IT",
        "question": "How can I connect to the company VPN?",
        "answer": "Install the approved VPN application and sign in using your company credentials."
    },

    {
        "category": "IT",
        "question": "How do I report a computer or software problem?",
        "answer": "Create an IT support ticket describing the technical issue."
    },

    # PAYROLL FAQs
    {
        "category": "Payroll",
        "question": "When will my monthly salary be credited?",
        "answer": "Monthly salaries are generally credited on the last working day of the month."
    },

    {
        "category": "Payroll",
        "question": "How can I download my salary slip?",
        "answer": "You can download your monthly payslip from the employee payroll portal."
    },

    {
        "category": "Payroll",
        "question": "What should I do if my salary is incorrect?",
        "answer": "Contact the payroll department with your employee details and salary information."
    },

    {
        "category": "Payroll",
        "question": "How can I update my bank account details for salary?",
        "answer": "Submit updated bank account information through the payroll portal."
    }
]


# Convert FAQs into DataFrame
faq_df = pd.DataFrame(faqs)

print("\n========== FAQ DATASET ==========")

print(faq_df[["category", "question"]])

print("\nTotal FAQs:", len(faq_df))

# Save FAQ dataset
faq_df.to_csv("faqs.csv", index=False)


# ============================================================
# STEP 2: LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

print("\n========== LOADING EMBEDDING MODEL ==========")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# ============================================================
# STEP 3: GENERATE FAQ EMBEDDINGS
# ============================================================

questions = faq_df["question"].tolist()

# Convert FAQ questions into embeddings
faq_embeddings = model.encode(
    questions,
    convert_to_numpy=True,
    normalize_embeddings=False
)

print("\n========== FAQ EMBEDDINGS ==========")

print("Number of FAQ embeddings:", len(faq_embeddings))

print("Embedding shape:", faq_embeddings.shape)

print("\nFirst FAQ embedding:")
print(faq_embeddings[0][:10])


# ============================================================
# STEP 4: IMPLEMENT COSINE SIMILARITY USING NUMPY
# ============================================================

def cosine_similarity(a, b):

    """
    Calculate cosine similarity between two vectors.

    Formula:
    cosine similarity = (A . B) / (||A|| * ||B||)
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    dot_product = np.dot(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)

    return float(similarity)


print("\n========== COSINE SIMILARITY TEST ==========")

similarity_test = cosine_similarity(
    faq_embeddings[0],
    faq_embeddings[1]
)

print("Similarity between first two FAQs:",
      round(similarity_test, 4))


# ============================================================
# STEP 5: IMPLEMENT SEMANTIC SEARCH
# ============================================================

def semantic_search(query, top_k=3):

    """
    Search FAQs based on semantic meaning.

    1. Convert query into embedding.
    2. Calculate similarity with FAQ embeddings.
    3. Sort similarity scores.
    4. Return top K matching FAQs.
    """

    # Convert user query into embedding
    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    scores = []

    # Calculate cosine similarity with each FAQ
    for index, faq_embedding in enumerate(faq_embeddings):

        score = cosine_similarity(
            query_embedding,
            faq_embedding
        )

        scores.append({
            "index": index,
            "similarity": score
        })

    # Sort by similarity score
    scores = sorted(
        scores,
        key=lambda x: x["similarity"],
        reverse=True
    )

    # Select top K results
    results = []

    for item in scores[:top_k]:

        index = item["index"]

        results.append({
            "category": faq_df.iloc[index]["category"],
            "question": faq_df.iloc[index]["question"],
            "answer": faq_df.iloc[index]["answer"],
            "similarity": round(item["similarity"], 4)
        })

    return results


# ============================================================
# STEP 6: IMPLEMENT KEYWORD SEARCH
# ============================================================

def keyword_search(query):

    """
    Search FAQs using exact keyword matching.

    The function checks how many query words
    appear in each FAQ question.
    """

    query_words = set(
        query.lower().split()
    )

    results = []

    for index, row in faq_df.iterrows():

        faq_words = set(
            row["question"].lower().split()
        )

        # Count common words
        matching_words = query_words.intersection(
            faq_words
        )

        score = len(matching_words)

        if score > 0:

            results.append({
                "category": row["category"],
                "question": row["question"],
                "answer": row["answer"],
                "keyword_score": score
            })

    # Sort by number of matching words
    results = sorted(
        results,
        key=lambda x: x["keyword_score"],
        reverse=True
    )

    return results


# ============================================================
# STEP 7: TEST REQUIRED QUERIES
# ============================================================

test_queries = [

    "I cannot log in",

    "when is salary credited",

    "work from home rules"

]

print("\n========== SEMANTIC SEARCH RESULTS ==========")

for query in test_queries:

    print("\nUser Query:", query)

    results = semantic_search(
        query,
        top_k=3
    )

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")

        print("Category:", result["category"])

        print("Question:", result["question"])

        print("Answer:", result["answer"])

        print("Similarity:", result["similarity"])


# ============================================================
# STEP 8: COMPARE SEMANTIC SEARCH AND KEYWORD SEARCH
# ============================================================

comparison_data = []

for query in test_queries:

    # Keyword search
    keyword_results = keyword_search(query)

    # Semantic search
    semantic_results = semantic_search(
        query,
        top_k=3
    )

    # Best keyword result
    if keyword_results:

        keyword_result = keyword_results[0]["question"]

    else:

        keyword_result = "No matching keyword found"

    # Best semantic result
    if semantic_results:

        semantic_result = semantic_results[0]["question"]

    else:

        semantic_result = "No semantic result found"

    comparison_data.append({

        "Query": query,

        "Keyword result": keyword_result,

        "Semantic result": semantic_result

    })


comparison_df = pd.DataFrame(comparison_data)

print("\n========== SEARCH COMPARISON TABLE ==========")

print(
    comparison_df.to_string(index=False)
)

comparison_df.to_csv(
    "search_comparison.csv",
    index=False
)


# ============================================================
# STEP 9: IDENTIFY TWO EXAMPLES WHERE SEMANTIC SEARCH WINS
# ============================================================

print("\n========== SEMANTIC SEARCH ADVANTAGES ==========")

examples = [
    {
        "query": "work from home rules",
        "explanation": (
            "Semantic search understands that 'rules' and "
            "'policies' have similar meanings. It can retrieve "
            "the HR work-from-home FAQ even though the exact "
            "words are different."
        )
    },

    {
        "query": "when is salary credited",
        "explanation": (
            "Semantic search understands the meaning of salary "
            "crediting and can match it with a question about "
            "when monthly salary is deposited, even when the "
            "wording differs."
        )
    }
]

for example in examples:

    print("\nQuery:", example["query"])

    print("Explanation:", example["explanation"])


# ============================================================
# STEP 10: SAVE EMBEDDINGS
# ============================================================

np.save(
    "faq_embeddings.npy",
    faq_embeddings
)

print("\n========== GENERATED FILES ==========")

print("1. faqs.csv")
print("2. faq_embeddings.npy")
print("3. search_comparison.csv")

print("\nTASK 2 COMPLETED SUCCESSFULLY!")


# ============================================================
# STEP 11: INTERACTIVE SEARCH
# ============================================================

print("\n========== TRY YOUR OWN SEARCH ==========")

while True:

    user_query = input(
        "\nEnter your question (or type exit): "
    )

    if user_query.lower() == "exit":
        print("Search ended.")
        break

    print("\nSemantic Search Results:")

    results = semantic_search(
        user_query,
        top_k=3
    )

    for result in results:

        print("\nQuestion:", result["question"])

        print("Category:", result["category"])

        print("Answer:", result["answer"])

        print("Similarity:", result["similarity"])
