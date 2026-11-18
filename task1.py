
# ============================================================
# TASK 1: DATA AUDIT AND MODEL EVALUATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.cluster import KMeans

np.random.seed(42)

CATEGORIES = ["Billing", "Technical", "Account"]


# ============================================================
# STEP 1: CREATE RAW DATASET (46 ROWS)
# ============================================================

tickets = [

    # BILLING - 15
    ("My payment was deducted twice from my account", "Billing"),
    ("I was charged twice for my subscription", "Billing"),
    ("Please refund my duplicate payment", "Billing"),
    ("My invoice shows an incorrect amount", "Billing"),
    ("I cannot download my monthly invoice", "Billing"),
    ("My credit card payment failed", "Billing"),
    ("I need a refund for my last transaction", "Billing"),
    ("My subscription renewal charged me twice", "Billing"),
    ("There is an unexpected charge on my bill", "Billing"),
    ("Please update my billing information", "Billing"),
    ("I was charged for a cancelled subscription", "Billing"),
    ("My payment is still pending", "Billing"),
    ("The invoice amount is incorrect", "Billing"),
    ("I need help understanding my subscription fees", "Billing"),
    ("I received a payment confirmation but no invoice", "Billing"),

    # TECHNICAL - 12
    ("My application crashes when I open it", "Technical"),
    ("The website keeps showing an error message", "Technical"),
    ("I cannot upload files to the platform", "Technical"),
    ("The dashboard is loading very slowly", "Technical"),
    ("I am unable to connect to the server", "Technical"),
    ("The mobile app freezes after login", "Technical"),
    ("I receive a 500 server error", "Technical"),
    ("The search function is not working", "Technical"),
    ("My browser displays a blank screen", "Technical"),
    ("The application crashes during file upload", "Technical"),
    ("I cannot access the API endpoint", "Technical"),
    ("The system keeps disconnecting", "Technical"),

    # ACCOUNT - 11
    ("I forgot my account password", "Account"),
    ("I cannot log in to my account", "Account"),
    ("Please help me reset my password", "Account"),
    ("My account has been locked", "Account"),
    ("I want to change my registered email address", "Account"),
    ("I cannot verify my account", "Account"),
    ("My profile information is incorrect", "Account"),
    ("I want to delete my account", "Account"),
    ("Someone changed my account password", "Account"),
    ("I am unable to access my user profile", "Account"),
    ("My account verification link has expired", "Account")
]

df = pd.DataFrame(tickets, columns=["text", "category"])

# Inconsistent casing
df.loc[2, "category"] = "BILLING"
df.loc[16, "category"] = "technical"
df.loc[30, "category"] = "ACCOUNT"
df.loc[35, "category"] = "billing"

# Three duplicate records
duplicates = df.iloc[[0, 15, 27]].copy()

df = pd.concat(
    [df, duplicates],
    ignore_index=True
)

# Five rows with missing values
missing_rows = pd.DataFrame({
    "text": [
        np.nan,
        "I need help with my account",
        np.nan,
        "My payment is not processing",
        np.nan
    ],
    "category": [
        "Billing",
        np.nan,
        "Technical",
        np.nan,
        "Account"
    ]
})

df = pd.concat(
    [df, missing_rows],
    ignore_index=True
)

df.to_csv("tickets_raw.csv", index=False)

print("\n========== RAW DATASET ==========")
print(df.head())
print("Rows before cleaning:", len(df))


# ============================================================
# STEP 2: DATA AUDIT
# ============================================================

print("\n========== DATA AUDIT ==========")

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Percentage:")
print((df.isnull().mean() * 100).round(2))

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nCategory Distribution:")
print(df["category"].value_counts(dropna=False))


# ============================================================
# STEP 3: DATA CLEANING
# ============================================================

before_rows = len(df)

clean_df = df.copy()

# Remove missing values
clean_df = clean_df.dropna(
    subset=["text", "category"]
)

# Standardize text
clean_df["text"] = clean_df["text"].str.strip()

# Standardize categories
clean_df["category"] = (
    clean_df["category"]
    .str.strip()
    .str.title()
)

# Remove duplicates
clean_df = clean_df.drop_duplicates()

# Remove empty text
clean_df = clean_df[
    clean_df["text"] != ""
]

# Keep valid categories
clean_df = clean_df[
    clean_df["category"].isin(CATEGORIES)
]

clean_df = clean_df.reset_index(drop=True)

after_rows = len(clean_df)

clean_df.to_csv("tickets_clean.csv", index=False)

print("\n========== CLEANING SUMMARY ==========")

print("Rows before cleaning:", before_rows)
print("Rows after cleaning:", after_rows)
print("Rows removed:", before_rows - after_rows)

print("\nMissing Values After Cleaning:")
print(clean_df.isnull().sum())

print("\nDuplicates After Cleaning:")
print(clean_df.duplicated().sum())

print("\nCategory Distribution:")
print(clean_df["category"].value_counts())

assert before_rows == 46
assert after_rows == 38


# ============================================================
# STEP 4: TRAIN TEST SPLIT (80/20)
# ============================================================

X = clean_df["text"]
y = clean_df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# STEP 5: BASELINE MODEL
# ============================================================

most_common_category = y_train.mode()[0]

baseline_predictions = np.repeat(
    most_common_category,
    len(y_test)
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    average="macro",
    zero_division=0
)

print("\n========== BASELINE MODEL ==========")

print("Most Common Category:", most_common_category)
print("Baseline Accuracy:", round(baseline_accuracy, 2))
print("Baseline Macro F1:", round(baseline_f1, 2))

baseline_cm = confusion_matrix(
    y_test,
    baseline_predictions,
    labels=CATEGORIES
)

print("\nBaseline Confusion Matrix:")
print(baseline_cm)


# ============================================================
# STEP 6: TF-IDF + LOGISTIC REGRESSION
# GRID SEARCH WITH CROSS VALIDATION
# ============================================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            random_state=42
        )
    )
])

parameter_grid = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2)
    ],

    "tfidf__min_df": [
        1
    ],

    "tfidf__max_df": [
        0.95,
        1.0
    ],

    "classifier__C": [
        0.1,
        0.5,
        1,
        2,
        5,
        10,
        20
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True
)

# Train using training data only
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_

# Predict test data once
model_predictions = model.predict(X_test)

# Evaluation
model_accuracy = accuracy_score(
    y_test,
    model_predictions
)

model_f1 = f1_score(
    y_test,
    model_predictions,
    average="macro",
    zero_division=0
)

print("\n========== BEST MODEL ==========")

print("Best Parameters:")
print(grid_search.best_params_)

print(
    "Best Cross Validation Macro F1:",
    round(grid_search.best_score_, 2)
)

print("\n========== MODEL EVALUATION ==========")

print("Model Accuracy:", round(model_accuracy, 2))
print("Model Macro F1:", round(model_f1, 2))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        model_predictions,
        labels=CATEGORIES,
        zero_division=0
    )
)

model_cm = confusion_matrix(
    y_test,
    model_predictions,
    labels=CATEGORIES
)

print("\nModel Confusion Matrix:")
print(model_cm)


# ============================================================
# STEP 7: CONFUSION MATRIX VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5)
)

sns.heatmap(
    baseline_cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=CATEGORIES,
    yticklabels=CATEGORIES,
    ax=axes[0]
)

axes[0].set_title("Baseline Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(
    model_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CATEGORIES,
    yticklabels=CATEGORIES,
    ax=axes[1]
)

axes[1].set_title("TF-IDF Logistic Regression")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300)
plt.show()


# ============================================================
# STEP 8: THREE MISCLASSIFIED TICKETS
# ============================================================

results = pd.DataFrame({
    "text": X_test.values,
    "actual_category": y_test.values,
    "predicted_category": model_predictions
})

misclassified = results[
    results["actual_category"] !=
    results["predicted_category"]
].copy()

def explain_error(row):

    actual = row["actual_category"]
    predicted = row["predicted_category"]

    if actual == "Billing" or predicted == "Billing":
        return "Payment and subscription terms overlap."

    elif actual == "Account" or predicted == "Account":
        return "Login and account access terms overlap."

    return "General system and error terminology overlaps."

misclassified["reason"] = misclassified.apply(
    explain_error,
    axis=1
)

print("\n========== MISCLASSIFIED TICKETS ==========")

if len(misclassified) >= 3:
    print(
        misclassified[
            [
                "text",
                "actual_category",
                "predicted_category",
                "reason"
            ]
        ].head(3).to_string(index=False)
    )
else:
    print(misclassified.to_string(index=False))
    print("Fewer than 3 errors occurred.")

results.to_csv("model_predictions.csv", index=False)

misclassified.to_csv(
    "misclassified_tickets.csv",
    index=False
)


# ============================================================
# STEP 9: KMEANS CLUSTERING WITHOUT LABELS
# ============================================================

cluster_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

all_tfidf = cluster_vectorizer.fit_transform(
    clean_df["text"]
)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(all_tfidf)

cluster_results = clean_df.copy()

cluster_results["cluster"] = cluster_labels

print("\n========== KMEANS CLUSTERING ==========")

print("\nCluster Distribution:")
print(cluster_results["cluster"].value_counts())

cluster_comparison = pd.crosstab(
    cluster_results["cluster"],
    cluster_results["category"]
)

print("\nCluster vs Real Category Comparison:")
print(cluster_comparison)

print("\n========== THREE CLUSTER COMPARISON LINES ==========")

for cluster in sorted(cluster_results["cluster"].unique()):

    counts = (
        cluster_results[
            cluster_results["cluster"] == cluster
        ]["category"]
        .value_counts()
    )

    dominant = counts.index[0]

    percentage = counts.iloc[0] / counts.sum() * 100

    print(
        f"Cluster {cluster}: {percentage:.1f}% "
        f"of tickets belong to {dominant}."
    )

cluster_results.to_csv(
    "kmeans_cluster_results.csv",
    index=False
)


# ============================================================
# STEP 10: MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Baseline",
        "TF-IDF + Logistic Regression"
    ],

    "Accuracy": [
        baseline_accuracy,
        model_accuracy
    ],

    "Macro F1": [
        baseline_f1,
        model_f1
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison.round(2).to_string(index=False))

comparison.to_csv(
    "model_comparison.csv",
    index=False
)


# ============================================================
# STEP 11: FINAL SUMMARY
# ============================================================

print("\n========== FINAL PROJECT SUMMARY ==========")

print("Rows before cleaning:", before_rows)
print("Rows after cleaning:", after_rows)

print("Baseline Accuracy:", round(baseline_accuracy, 2))
print("Model Accuracy:", round(model_accuracy, 2))
print("Model Macro F1:", round(model_f1, 2))

print("\n========== GENERATED FILES ==========")

files = [
    "tickets_raw.csv",
    "tickets_clean.csv",
    "model_predictions.csv",
    "misclassified_tickets.csv",
    "kmeans_cluster_results.csv",
    "model_comparison.csv",
    "confusion_matrices.png"
]

for file in files:
    print(file)

print("\nTASK 1 COMPLETED SUCCESSFULLY!")
