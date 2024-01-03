from abc import ABC, abstractmethod
from typing import Any, List
from open_rqa.schema.document import Document


class BaseTextLoader(ABC):
    @abstractmethod
    def load_data(self, *args: Any, **kwargs: Any) -> List[Document]:
        raise NotImplementedError
    
    @abstractmethod
    def save_texts(self, texts: List[Document]):
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
