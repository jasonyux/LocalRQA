from typing import List
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import (
    LocalFileStore,
)
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.document import Document as LangChainDocument
from open_rqa.schema.document import Document
from open_rqa.retrievers.base import BaseRetriever, RetrievalOutput


def langchain_doc_to_open_rqa_doc(langchain_doc: LangChainDocument):
    return Document.from_dict({
        'page_content': langchain_doc.page_content,
        'metadata': langchain_doc.metadata
    })


class FaissRetriever(BaseRetriever):
    def __init__(
        self,
        texts: List[Document],
        embeddings=OpenAIEmbeddings(model='text-embedding-ada-002'),
        index_path="./index"
    ) -> None:
        """

        Args:
            texts (List[Document]): documents for retriever
            embeddings (_type_, optional): embeddings wrapper supported by LangChain. Defaults to OpenAIEmbeddings(model='text-embedding-ada-002').
            index_path (str, optional): saving index path. Defaults to "./index".
        """
        
        fs = LocalFileStore(index_path)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            embeddings, fs, namespace="customized"
        )
        super().__init__(texts, cached_embedder)
        self.retriever = self._init_retriever()
        return
    
    def _init_retriever(self, **kwargs):
        """initiate FAISS retriever

        Returns:
            _type_: retriever
        """

        docsearch = FAISS.from_documents(self.texts, self.embeddings)
        
        retriever = docsearch.as_retriever(**kwargs)
        return retriever
    
    def retrieve(self, batch_questions: List[str]) -> RetrievalOutput:
        """given a batched query and dialogue history, retrieve relevant documents

        Args:
            batch_questions (List[str]): _description_

        Returns:
            RetrievalOutput: _description_
        """
        all_docs = []
        for query in batch_questions:
            docs = self.retriever.get_relevant_documents(query)
            docs = [langchain_doc_to_open_rqa_doc(doc) for doc in docs]
            all_docs.append(docs)

        output = RetrievalOutput(
            batch_source_documents=all_docs
        )
        return output
