"""

Utility script for inspecting individual HotpotQA dev examples.
Given a question ID (or an index), it retrieves the evidence returned by the
retriever and prints the question, gold answer, model prediction, and the
top-k retrieved paragraphs. Used to analyse success/failure cases.
"""



import json
from retrieve_data import data, Retriever
import qa_engine

# Load predictions
with open("dev_fullwiki_pred.json", "r", encoding="utf-8") as f:
    preds = json.load(f)["answer"]

def inspect(idx):
    item = data[idx]

    q_id = item["_id"]
    question = item["question"]
    gold_answer = item["answer"]
    predicted_answer = preds.get(q_id, "[MISSING]")

    # Retrieve evidence using your retriever (with the same k)
    retriever = Retriever(k=5, rerank=True)
    evidence = retriever.retrieve_top_k(question, data)

    print("\n==============================")
    print(f"Example #{idx}")
    print(f"Question ID: {q_id}")
    print(f"QUESTION: {question}")
    print(f"GOLD ANSWER: {gold_answer}")
    print(f"PREDICTED ANSWER: {predicted_answer}")
    print("\n Retrieved Evidence ---")
    print(evidence)
    print("==============================\n")


# Change the index to any number 0–7404 to inspect, 
inspect(1)
