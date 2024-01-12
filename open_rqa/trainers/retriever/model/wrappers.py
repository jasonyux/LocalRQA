from langchain.vectorstores.faiss import FAISS

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from open_rqa.retrievers.base import RetrievalOutput
from open_rqa.schema.dialogue import RQAOutput
from open_rqa.schema.document import Document
from open_rqa.trainers.retriever.embeddings import LocalEmbeddings
import torch
import logging

logger = logging.getLogger(__name__)


class RetrievalModel(ABC):
    embeddings = None

    def __init__(self, *args, **kwargs):
        return
    
    @abstractmethod
    def build_index(self, documents: List[Document] = [], format_str='title: {title} content: {text}') -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def retrieve(self, batch_inputs, documents=None, indices=None) -> RetrievalOutput:
        raise NotImplementedError


class RetrieverfromBertModel(RetrievalModel):
    def __init__(self, model, tokenizer, search_kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.search_args = search_kwargs
        self.embeddings: LocalEmbeddings = None
        return
    
    def build_index(self, documents: List[Document] = [], format_str='title: {title} content: {text}') -> torch.Tensor:
        embeddings = LocalEmbeddings(self.model, self.tokenizer, index_path=None)

        # asuming we are doing an inner product search, then normalize_L2=False
        normalize_L2 = self.search_args['search_kwargs'].pop('normalize_L2', False)
        docsearch = FAISS.from_documents(documents, embeddings)  # this will call from_texts under the hood
        self.embeddings = embeddings
        
        # self.retriever = docsearch.as_retriever(**self.search_args)
        self.retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={'k': 4})
        embedding_tensor = torch.tensor(self.embeddings.document_embeddings)
        return embedding_tensor
    
    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        docs = self.retriever.get_relevant_documents(question)
        # convert langchain Doc to our Doc
        docs = [Document.from_langchain_doc(doc) for doc in docs]
        return docs
    
    def _batch_get_docs(self, questions: List[str], inputs: Dict[str, Any]) -> List[List[Document]]:
        all_docs = []
        for question in questions:
            docs = self._get_docs(question, inputs)
            all_docs.append(docs)
        return all_docs
    
    def retrieve(self, batch_inputs, passages=None, indices=None) -> RetrievalOutput:
        questions = batch_inputs["question"]
        chat_history_strs = batch_inputs["chat_history_str"]

        # retrieve
        docs = self._batch_get_docs(questions, {})
        output = RetrievalOutput(
            batch_source_documents=docs
        )
        return output
