from typing import List
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import (
    LocalFileStore,
)
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from open_rqa.schema.document import Document
from open_rqa.retrievers.base import BaseRetriever, RetrievalOutput
from copy import deepcopy


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
        docs = self.prepare_docs_for_retrieval(texts)

        super().__init__(docs, cached_embedder)
        self.retriever = self._init_retriever()
        return

    def prepare_docs_for_retrieval(self, texts: List[Document]):
        """prepare documents so that LangChain FAISS would recognize it, and embed using our fmt_content

        Args:
            texts (List[Document]): _description_

        Returns:
            _type_: _description_
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
