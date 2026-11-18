
import tiktoken
import pandas as pd

# ==========================================
# TASK 3: TOKENS, COST, TEMPERATURE
# AND HALLUCINATION
# NO API PAYMENT REQUIRED
# ==========================================

MODEL = "gpt-4o-mini"

# ==========================================
# EXPERIMENT 1: TOKEN COUNTING
# ==========================================

texts = [
    "Artificial intelligence is changing the world.",
    "Python is a popular programming language used for AI.",
    "Machine learning helps computers learn from data.",
    "ChatGPT is an AI chatbot developed by OpenAI."
]

encoding = tiktoken.encoding_for_model(MODEL)

token_results = []

for i, text in enumerate(texts, start=1):

    tokens = encoding.encode(text)
    token_count = len(tokens)
    words = len(text.split())

    words_per_token = words / token_count

    token_results.append({
        "Text": f"Text {i}",
        "Words": words,
        "Tokens": token_count,
        "Words per Token": round(words_per_token, 3)
    })

print("\n===== EXPERIMENT 1: TOKEN COUNTING =====")

df1 = pd.DataFrame(token_results)
print(df1.to_string(index=False))


# ==========================================
# EXPERIMENT 2: MONTHLY API COST
# ==========================================

# Example prices per million tokens.
# Replace with your trainer's prices.

INPUT_PRICE = 0.15
OUTPUT_PRICE = 0.60

CHATS_PER_DAY = 300
DAYS_PER_MONTH = 30

INPUT_TOKENS = 1200
OUTPUT_TOKENS = 300

total_chats = CHATS_PER_DAY * DAYS_PER_MONTH

total_input_tokens = total_chats * INPUT_TOKENS
total_output_tokens = total_chats * OUTPUT_TOKENS

input_cost = (
    total_input_tokens / 1_000_000
) * INPUT_PRICE

output_cost = (
    total_output_tokens / 1_000_000
) * OUTPUT_PRICE

monthly_cost = input_cost + output_cost

cost_results = [
    {"Metric": "Chats per month", "Value": total_chats},
    {"Metric": "Input tokens", "Value": total_input_tokens},
    {"Metric": "Output tokens", "Value": total_output_tokens},
    {"Metric": "Input cost USD", "Value": round(input_cost, 4)},
    {"Metric": "Output cost USD", "Value": round(output_cost, 4)},
    {"Metric": "Monthly total USD", "Value": round(monthly_cost, 4)}
]

print("\n===== EXPERIMENT 2: MONTHLY COST =====")

df2 = pd.DataFrame(cost_results)
print(df2.to_string(index=False))


# ==========================================
# EXPERIMENT 3: TEMPERATURE COMPARISON
# ==========================================

prompt = "Explain the importance of education."

temperature_results = []

# Illustrative responses, not live model outputs.

responses_temp_0 = [
    "Education provides knowledge and develops skills.",
    "Education provides knowledge and develops skills.",
    "Education provides knowledge and develops skills.",
    "Education provides knowledge and develops skills.",
    "Education provides knowledge and develops skills."
]

responses_temp_1 = [
    "Education opens doors to opportunities and success.",
    "Learning helps people become creative problem solvers.",
    "Education builds confidence and improves society.",
    "Schools help students discover their talents.",
    "Knowledge empowers people to shape their future."
]

for run in range(5):

    temperature_results.append({
        "Temperature": 0,
        "Run": run + 1,
        "Response": responses_temp_0[run],
        "Type": "Illustrative demonstration"
    })

    temperature_results.append({
        "Temperature": 1.0,
        "Run": run + 1,
        "Response": responses_temp_1[run],
        "Type": "Illustrative demonstration"
    })

print("\n===== EXPERIMENT 3: TEMPERATURE =====")

df3 = pd.DataFrame(temperature_results)
print(df3.to_string(index=False))
# ==========================================
# EXPERIMENT 4: HALLUCINATION TEST
# ==========================================
made_up_question = (
    "What year did the International Moonlight "
    "Programming Championship first take place, "
    "and who won it?"
)

hallucination_results = [
    {
        "Test": "Without uncertainty instruction",
        "Question": made_up_question,
        "Response": (
            "Illustrative example: The model might invent "
            "a year and a winner for this fictional event."
        ),
        "Observation": "Potential hallucination"
    },
    {
        "Test": "With uncertainty instruction",
        "Question": (
            made_up_question +
            " Say I don't know if unsure."
        ),
        "Response": (
            "Illustrative example: I don't know. "
            "I cannot verify that this event exists."
        ),
        "Observation": "Acknowledges uncertainty"
    }
]

print("\n===== EXPERIMENT 4: HALLUCINATION =====")

df4 = pd.DataFrame(hallucination_results)
print(df4.to_string(index=False))


# ==========================================
# SAVE ALL RESULTS
# ==========================================

df1.to_csv("token_results.csv", index=False)
df2.to_csv("cost_results.csv", index=False)
df3.to_csv("temperature_results.csv", index=False)
df4.to_csv("hallucination_results.csv", index=False)

print("\n===================================")
print("ALL FOUR EXPERIMENTS COMPLETED!")
print("===================================")

print("\nFiles created:")
print("1. token_results.csv")
print("2. cost_results.csv")
print("3. temperature_results.csv")
print("4. hallucination_results.csv")

print("\nNo OpenAI API credits were used.")
