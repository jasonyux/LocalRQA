from abc import ABC, abstractmethod
from typing import Any, List
from local_rqa.schema.document import Document


class BaseTextLoader(ABC):
    @abstractmethod
    def load_data(self, *args: Any, **kwargs: Any) -> List[Document]:
        """Load into Document

        Raises:
            NotImplementedError

        Returns:
            List[Document]
        """
        raise NotImplementedError
    
    @abstractmethod
    def save_texts(self, texts: List[Document]):
        """Save the chunked documents into the pickle file

        Args:
            texts (List[Document])

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
    
    def _convert_doc(self, texts):
        all_docs = []
        for text in texts:
            new_doc = Document(
                page_content=text.page_content,
                metadata=text.metadata,
                )
            all_docs.append(new_doc)
        return all_docs
