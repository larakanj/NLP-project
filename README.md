# NLP-project: Hybrid HotpotQA Pipeline

This repository contains the implementation of a hybrid open-domain question answering (QA) system developed for the HotpotQA Fullwiki setting as part of the NLP coursework.
The system combines BM25 retrieval, MiniLM reranking, and Mistral-based answer generation, operating entirely in a zero-shot setting (no supervised fine-tuning).

The project includes scripts for generating predictions, inspecting examples for case studies, and evaluating the output using the official HotpotQA evaluation script.


## Repository Structure
### Core Pipeline Files

`retrieve_data.py`

Implements the retrieval component of the hybrid pipeline.

- Uses BM25 to select top-k candidate paragraphs for each question.
- Optionally reranks them using MiniLM sentence embeddings.
- Produces a compact evidence text block passed to the LLM.

`qa_engine.py`

Handles answer generation using the Mistral API.

- Builds the structured reasoning prompt.
- Enforces “evidence-only” rule.
- Returns the final short answer for each question.

`generate_pred.py`

Runs the full pipeline on the HotpotQA dev (fullwiki) dataset.

- Retrieves evidence via the Retriever.
- Generates answers for all examples.
- Saves predictions in the official HotpotQA format:

```
{
    "answer": {id: predicted_answer},
    "sp": {id: []}  // supporting facts unused in this pipeline
}
```

This script used to produce the three prediction files used in the experiments.

## Evaluation and Analysis Tools

`hotpot_evaluate_v1.py`

The official HotpotQA evaluation script.

Computes:
- Exact Match (EM)
- F1
- Precision
- Recall

Supporting-fact metrics (zero for this system)

`inspect_example.py`

A utility script used for case study inspection.
Given an example ID (or index), it prints:
- The question
- Gold answer
- Model prediction
- The retrieved evidence paragraphs

This script was essential for analysing representative success/failure cases in the report.
It explains why the model predicted correctly or incorrectly based on what evidence was retrieved.


## Prediction Files (for Ablation Experiments)
Three prediction files are included to support the ablation study performed in the evaluation section of the coursework.

1. dev_fullwiki_pred.json
- Main prediction file used in the final evaluation.
- Generated with **k = 5 retrieved paragraphs**, which is the default setting.

2. dev_fullwiki_pred_k=0.json
- Prediction file generated with **k = 0**.
- The model receives *no retrieved evidence* and must rely on minimal context and LLM priors.
- Used to measure how much retrieval improves performance over a no-evidence baseline.

3. dev_fullwiki_pred_k=10.json
- Prediction file generated with **k = 10 retrieved paragraphs**.
- Tests whether retrieving more paragraphs improves or harms accuracy.
- Demonstrates the trade-off between relevant information and retrieval noise.

These three files allow comparison across retrieval depths and directly support the ablation study discussed in the report.

## How to Run the Pipeline 
1. Generate predictions
 
   ```
   python generate_pred.py
   ```

3. Evaluate
   
   ```
   python hotpot_evaluate_v1.py dev_fullwiki_pred.json hotpot_dev_fullwiki_v1.json
   ```

4. Inspect individual examples
   
    ```
   python inspect_example.py
    ```


## Notes
- The system is zero-shot (no supervised fine-tuning).

- Supporting facts (`sp`) are left empty intentionally, so supporting-fact scores are 0 by design.

- Retrieval is the main bottleneck, the analysis in the final report discusses this in detail.
