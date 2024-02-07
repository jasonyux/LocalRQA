from typing import List, Tuple
import re
import string

from rank_bm25 import BM25Okapi

from local_rqa.schema.document import Document
from local_rqa.retrievers.base import BaseRetriever, RetrievalOutput


def normalize_string(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class BM25Tokenizer:
    def __call__(self, text: str) -> List[str]:
        normalized_text = normalize_string(text)
        tokenized = re.split(r'\s|, |\?|\r|\n|\*', normalized_text)
        return [t for t in tokenized if t != '']


class BM25Retriever(BaseRetriever):
    def __init__(self, texts: List[Document]) -> None:
        self.bm25_tokenizer = BM25Tokenizer()
        formatted_texts = [doc.to_dict()['fmt_content'] for doc in texts]
        self.bm25 = BM25Okapi(formatted_texts, tokenizer=self.bm25_tokenizer)
        self.documents = texts

    def retrieve(self, batch_questions: List[str]) -> RetrievalOutput:
        """given a batched query and dialogue history, retrieve relevant documents

        Args:
            batch_questions (List[str]): _description_

        Returns:
            RetrievalOutput: _description_
        """
        all_docs = []
        for query in batch_questions:
            tokenized_query = self.bm25_tokenizer(query)
            docs = self.bm25.get_top_n(tokenized_query, self.documents, n=4)
            all_docs.append(docs)

        output = RetrievalOutput(
            batch_source_documents=all_docs
        )
        return output