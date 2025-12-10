"""

This script runs the full hybrid QA pipeline on the HotpotQA dev (fullwiki) set.
For each question, it retrieves evidence using BM25 + MiniLM reranking and then
uses the Mistral model to generate an answer. Outputs are saved in the official
HotpotQA prediction format (answer + empty supporting facts).
"""


from retrieve_data import Retriever, data
import qa_engine
import json
import time

OUTPUT_FILE = "dev_fullwiki_pred_k.json"

def main():
    retriever = Retriever(k=0, rerank=True)
    answers = {}
    sps = {}

    total = len(data)
    print(f" Total examples to process: {total}")

    start_all = time.time()
    for i, item in enumerate(data):
        if "_id" in item:
            q_id = item["_id"]
        else:
            item.get("id", str(i))

        question = item["question"]

        try:
            evidence_text = retriever.retrieve_top_k(question,data)

        except Exception as e:
            print(f"[Retriever error] id={q_id}: {e}")
            evidence_text = ""

        # call LLM to generate an answer
        try: 
            answer = qa_engine.generate_answer(question, evidence_text)

        except Exception as e:
            print(f'[llm error] id={q_id}: {e}')
            answer=''

        # strip unwated whitespace
        answer = answer.strip()

        # savee answer
        answers[q_id]= answer

        # supporting facts as empty list as pipeline doesn't support it
        sps[q_id]= []

        # process log
        if(i+1) % 50 == 0 or i<3:
            elapsed = time.time() - start_all
            print(f"[{i+1}/{total}] id={q_id}  | ans len={len(answer):3} | elapsed={elapsed:.1f}s")


    # final json

    out = {'answer': answers, 'sp': sps}
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        json.dump(out, fout, ensure_ascii=False, indent=2)

    print(f'saved prediction to {OUTPUT_FILE}')

if __name__ == "__main__":
    print('start predicition')
    main()

    print('done with prediction')