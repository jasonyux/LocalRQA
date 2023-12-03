from dataclasses import dataclass, field
from typing import Callable


def default_document_formatter(document):
    """converts a Document object to a string, utilizing the metadata and content fields

    Args:
        document (_type_): _description_

    Returns:
        _type_: _description_
    """
    formatted_str = ''
    for key, value in document.metadata.items():
        formatted_str += f'{key}: {value}\n'
    formatted_str += 'Content:\n'
    formatted_str += document.content
    return formatted_str.strip()


@dataclass
class Document:
    """representing a chunk of text, along with its metadata (e.g. title, author, url, etc.)
    """
    title: str  # title, subtitle, url, etc. This is REQUIRED for RQA prompts
    content: str
    to_string: Callable = default_document_formatter
    metadata: dict = field(default_factory=dict)