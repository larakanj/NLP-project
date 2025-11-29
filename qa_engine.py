import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)


# prompt template
_PROMPT = """You’re a really smart system that is very good at answering questions that need more than one piece of information. 
Your whole thing is understanding questions that have several steps, figuring out which bits of information 
you need to extract, and then linking everything together properly to get to the final answer.

Before doing anything, please start by carefully reading all of the EVIDENCE below. 
When I say carefully, I really mean read it slowly and make sure you understand it, 
and most importantly keep all the important details in your memory because everything you’re going to say MUST come only from that evidence you just read. 
Basically: no outside knowledge, no guessing, no “general facts”. Everything has to be grounded in what you read.

Once you’ve read the evidence and saved it in your memory, I want you to think step by step internally 
(don’t show me your steps). Break down the question in your mind into small tasks. To do so, when you carefully read the question, every time you see that you need to extract an information, that will be considered as a task. So for each task you do, you only ever have to focus on one thing at a time.
For example, identify what things you need to find, what parts of the evidence relate to each part of the question, 
and what you need to understand first in order to answer the rest. 
Just make it simple for yourself: each small task should represent ONE (and only one) thing that you need to extract.

Once you have that list of "small" tasks you have to do step by step, go through those small tasks one by one (still internally), and answer them using ONLY the information from the evidence as we said. 
Do not bring anything from outside. If something is not mentioned in the evidence, treat it as unknown. 
But if it is mentioned, rely on it properly.

After you’ve done all your internal steps and reached the final answer, I want you to return ONLY the final answer. 
Your answer should be a short, clear, simple, concise, direct response.
Don’t explain anything or show your reasoning.

Here are formatting rules:
- If the answer is yes or no, return exactly "yes" or "no".
- If the answer is a name, title, place, or any entity, return ONLY that phrase without any extra words.
- If the answer is a number, year, or date, return the cleanest version of it (just the value).
- No extra text, no introductions, no explanations.

EVIDENCE:
{EVIDENCE}

QUESTION:
{QUESTION}

FINAL ANSWER:
"""

def build_prompt(question: str, evidence_text: str) -> str:
    #Fill the template with the retrieved evidence and the question.
    return _PROMPT.format(EVIDENCE=evidence_text, QUESTION=question)

def generate_answer(question: str, evidence_text: str, model: str = "mistral-small-latest") -> str:

    prompt = build_prompt(question, evidence_text)
    messages = [{"role": "user", "content": prompt}]
    resp = client.chat.complete(model=model, messages=messages)
    text = resp.choices[0].message.content.strip()

    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return text
