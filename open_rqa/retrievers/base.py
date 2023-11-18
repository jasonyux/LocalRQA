from typing import List
from abc import ABC, abstractmethod
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.schema.document import Document


class RetrievalOutput(ABC):
    source_documents: List[List[Document]]


class BaseRetriever(ABC):
    """retrieves relevant documents from a corpus given a query
    """
    @abstractmethod
    def retrieve(
        self,
        batch_query: List[str],
        batch_dialogue_history: List[DialogueSession],
    ) -> RetrievalOutput:
        """given a batched query and dialogue history, retrieve relevant documents (potentially rephrasing the query)

        Args:
            batch_query (List[str]): _description_
            batch_dialogue_history (List[DialogueSession]): _description_

        Returns:
            RetrievalOutput: _description_
        """
        raise NotImplementedError
