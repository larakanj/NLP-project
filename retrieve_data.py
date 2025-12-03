import json
from rank_bm25 import BM25Okapi
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
from sentence_transformers import util

file_path = "hotpot_dev_fullwiki_v1.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total examples: {len(data)}\n")

class Retriever:
    """
    Retriever class for paragraph retrieval from a list of context paragraphs.
    
    """

    def __init__(self,k=10,rerank=True):
        """
        Intialise the Retriever.

        Args:
            k (int): Number of relevant paragraphs to retrieve.
        
        """
        self.k = k
        self.bm25 = None
        self.rerank = rerank
        if self.rerank:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def find_context(self,question,data):
        """
        Find the matching context for a question.

        Args:
            question (str): The input question
            data: HotpotQA dataset
        
        Returns:
            list/None: Context for the question, or None if not found.
        
        """
        for item in data:
            if item["question"] == question:
                return item["context"]
        return None
    
    
    def create_docs(self,context):
        """
        Combine the context paragraphs into a document.

        Args:
            context: List of [title,text]
        
        Returns:
            list[str]: Combined paragraphs in a document.
        
        """

        document_pool = []
        for title,text in context:
            paragraph = " ".join(text).strip()
            document_pool.append(paragraph)
        
        return document_pool

    def retrieve_top_k(self,question,data):

        """
        Retrieve top-k relevant paragraphs for a question.

        Args:
            question(str): Input question
            context: List of [title,text]
        
        Returns:
            str: Combined top-k paragraphs
        
        """

        context = self.find_context(question,data)

        document_pool = self.create_docs(context)

        tokenized_documents = [document.split() for document in document_pool]
        self.bm25 = BM25Okapi(tokenized_documents)

        scores = self.bm25.get_scores(question)
        top_k = scores.argsort()[::-1][:self.k]

        paragraphs = [document_pool[i] for i in top_k]

        top_k_rerank = 5

        if self.rerank and len(paragraphs) > 1:
            paragraph_embeddings = self.model.encode(paragraphs, convert_to_numpy=True)
            question_embedding = self.model.encode([question], convert_to_numpy=True)
            similarities = util.cos_sim(paragraph_embeddings, question_embedding)[:,0]
            reranked_indices = similarities.argsort(descending=True)
            paragraphs = [paragraphs[i] for i in reranked_indices[:top_k_rerank]]

        return "\n\n".join(paragraphs)


#print(item["question"])
#print(item["context"])
#example use
item = data[8]
retriever = Retriever()
result = retriever.retrieve_top_k(item["question"],data)
print(result)





