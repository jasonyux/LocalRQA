from langchain.schema.document import Document as LangChainDocument
from dataclasses import dataclass, field
from typing import Dict, Any
from copy import deepcopy


def default_document_formatter(document):
    """converts a Document object to a string, utilizing the metadata and content fields

    Args:
        document (_type_): _description_

    Returns:
        _type_: _description_
    """
    formatted_str = ''
    if 'source' in document.metadata:
        formatted_str += f'Source: {document.metadata["source"]}\n'
    if 'title' in document.metadata:
        formatted_str += f'Title: {document.metadata["title"]}\n'
    formatted_str += 'Content:\n'
    formatted_str += document.page_content
    return formatted_str.strip()


@dataclass
class Document:
    """representing a chunk of text, along with its metadata (e.g. title, author, url, etc.)
    """
    page_content: str
    fmt_content: str = field(default='')  # content formatted with metadata information
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.fmt_content == '':
            self.fmt_content = default_document_formatter(self)
        return

    def to_dict(self) -> Dict[str, Any]:
        """converts the Document object into a dictionary

        Returns:
            dict[str, Any]: _description_
        """
        # return asdict(self)  # this also encodes the to_string function
        return {
            'page_content': self.page_content,
            'fmt_content': self.fmt_content,
            'metadata': self.metadata
        }

    @staticmethod
    def from_dict(document_dict):
        """converts a dictionary to a Document object.

        Args:
            document_dict (_type_): _description_
        """
        document = Document(
            page_content=document_dict['page_content'],
            fmt_content=document_dict.get('fmt_content', ''),
            metadata=document_dict.get('metadata', {})
        )
        return document

    @staticmethod
    def from_langchain_doc(document: LangChainDocument):
        """converts a langchain Document object to our Document object

        Args:
            document (_type_): _description_

        Returns:
            Document: _description_
        """
        return Document.from_dict({
            'page_content': document.page_content,
            'metadata': document.metadata
        })

    def to_langchain_doc(self) -> LangChainDocument:
        """converts our Document object to a langchain Document object

        Returns:
            _type_: _description_
        """
        metadata = deepcopy(self.metadata)
        if 'fmt_content' not in metadata:
            metadata['fmt_content'] = self.fmt_content  # add the extra field into metadata
        return LangChainDocument(
            page_content=self.page_content,
            metadata=metadata
        )

    def clone(self):
        """clone the Document object

        Returns:
            Document: cloned Document object
        """
        cloned_document = Document(
            page_content=self.page_content,
            fmt_content=self.fmt_content,
            metadata=deepcopy(self.metadata)
        )
        return cloned_document