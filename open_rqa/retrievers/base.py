from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod
from open_rqa.base import Component
from open_rqa.schema.document import Document


@dataclass
class RetrievalOutput(ABC):
    batch_source_documents: List[List[Document]]


class BaseRetriever(Component):
    """retrieves relevant documents from a corpus given a query
    """
    run_input_keys = ["batch_questions"]

    def __init__(self, texts: List[Document], embeddings) -> None:
        super().__init__()
        self.texts = texts
        self.embeddings = embeddings

    @abstractmethod
    def retrieve(
        self,
        batch_questions: List[str],
    ) -> RetrievalOutput:
        """given a batched query, retrieve relevant documents (query rephrasing is handled by the RQA pipeline)

        Args:
            batch_questions (List[str]): _description_

        Returns:
            RetrievalOutput: _description_
        """
        raise NotImplementedError

    def run(self, *args, **kwargs):
        return self.retrieve(*args, **kwargs)


class DummyRetriever(BaseRetriever):
    """a mock retriever used for testing

    Args:
        BaseRetriever (_type_): _description_
    """
    def retrieve(
        self,
        batch_questions: List[str],
    ) -> RetrievalOutput:
        dummy_document = Document(
            title="dummy title",
            content="dummy document",
            metadata={"dummy": "dummy"},
        )
        return RetrievalOutput(
            batch_source_documents=[[dummy_document] for _ in batch_questions]
        )
