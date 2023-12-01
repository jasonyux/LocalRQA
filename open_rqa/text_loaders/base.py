from abc import ABC, abstractmethod
from typing import Any, List
from langchain.schema import Document


class BaseTextLoader(ABC):
    @abstractmethod
    def load_data(self, *args: Any, **kwargs: Any) -> List[Document]:
        raise NotImplementedError
    
    @abstractmethod
    def save_texts(self, texts: List[Document]):
        raise NotImplementedError
