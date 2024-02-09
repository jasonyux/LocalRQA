from typing import List, Optional
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import (
    LocalFileStore,
)
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from local_rqa.schema.document import Document
from local_rqa.retrievers.base import BaseRetriever, RetrievalOutput
from copy import deepcopy
import logging
import pickle
import os


logger = logging.getLogger(__name__)


class FaissRetriever(BaseRetriever):
    def __init__(
        self,
        texts: List[Document],
        embeddings=None,
        index_path="./index",
        **kwargs
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
        self.docs = self.prepare_docs_for_retrieval(texts)

        super().__init__(self.docs, cached_embedder)
        self.retriever = self._init_retriever(**kwargs)
        return

    def prepare_docs_for_retrieval(self, texts: List[Document]):
        """prepare documents so that LangChain FAISS would recognize it, and embed using our fmt_content

        Args:
            texts (List[Document]): List of Document

        Returns:
            List[LangChainDocument]: List of Langchain Document
        """
        langhchain_docs = []
        for doc in texts:
            l_doc = doc.to_langchain_doc()
            # make sure we can recover this later
            l_doc.metadata['page_content'] = doc.page_content
            l_doc.metadata['fmt_content'] = doc.fmt_content
            l_doc.page_content = doc.fmt_content  # page_content is embedded
            langhchain_docs.append(l_doc)
        return langhchain_docs
    
    def _init_retriever(self, **kwargs):
        """initiate FAISS retriever

        Returns:
            _type_: retriever
        """
        # print('_init_retriever', self.texts[0].metadata.keys())
        docsearch = FAISS.from_documents(self.docs, self.embeddings)
        
        retriever = docsearch.as_retriever(**kwargs)
        return retriever
    
    def retrieve(self, batch_questions: List[str]) -> RetrievalOutput:
        all_docs = []
        for query in batch_questions:
            docs = self.retriever.get_relevant_documents(query)
            # convert back to ours
            our_docs = []
            for doc in docs:
                our_doc = Document.from_langchain_doc(doc)
                metadata = deepcopy(doc.metadata)
                our_doc.page_content = metadata.pop('page_content', None)
                our_doc.fmt_content = metadata.pop('fmt_content', None)
                our_doc.metadata = metadata
                our_docs.append(our_doc)
            
            all_docs.append(our_docs)

        output = RetrievalOutput(
            batch_source_documents=all_docs
        )
        return output
    
    def retrieve_w_score(self, batch_questions: List[str]):
        """_summary_

        Args:
            batch_questions (List[str]): List of questions

        Raises:
            ValueError: The retriever search_type must be "similarity"

        Returns:
            RetrievalOutput: List of Documents with score in the metadata
        """
        all_docs = []
        for query in batch_questions:
            if self.retriever.search_type != "similarity":
                raise ValueError(f"Only search_type='similarity' is supported with scores")
            k_value = 4 if not self.retriever.search_kwargs.get('k') else self.retriever.search_kwargs.get('k')
            docs_and_scores = self.retriever.vectorstore.similarity_search_with_score(query, k=k_value)
            for doc, score in docs_and_scores:
                doc.metadata = {**doc.metadata, **{"retrieve_score": float(score)}}
            docs = [doc for (doc, _) in docs_and_scores]
            # convert back to ours
            our_docs = []
            for doc in docs:
                our_doc = Document.from_langchain_doc(doc)
                metadata = deepcopy(doc.metadata)
                our_doc.page_content = metadata.pop('page_content', None)
                our_doc.fmt_content = metadata.pop('fmt_content', None)
                our_doc.metadata = metadata
                our_docs.append(our_doc)
            all_docs.append(our_docs)

        output = RetrievalOutput(
            batch_source_documents=all_docs
        )
        return output

    @staticmethod
    def from_disk(
        database_path: Optional[str] = None,
        document_path: Optional[str] = None,
        index_path: str = "./index",
        embeddings = None
    ):
        """Load document index from disk

        Args:
            database_path (Optional[str], optional): folder for the document and index files. Defaults to None.
            document_path (Optional[str], optional): document path. Defaults to None.
            index_path (str, optional): index path. Defaults to "./index".
            embeddings (_type_, optional): retriever embedding model. Defaults to None.

        Raises:
            ValueError: both database_path and document_path cannot be None

        Returns:
            FaissRetriever: faiss retriever
        """
        if database_path is not None:
            # assume the following filenames
            document_path = os.path.join(database_path, "documents.pkl")
            index_path = os.path.join(database_path, "index")
        elif document_path is None:
            raise ValueError("both database_path and document_path cannot be None")
        
        logger.info(f"Loaded documents from {document_path}")
        with open(document_path, 'rb') as fread:
            documents = pickle.load(fread)
        logger.info(f"Loaded {len(documents)} documents")

        return FaissRetriever(
            texts=documents,
            embeddings=embeddings,
            index_path=index_path
        )