from dataclasses import dataclass, field
from typing import Any, List
from open_rqa.schema.document import Document


@dataclass
class RQAOutput:
    batched_answers: List[str]
    batched_source_documents: List[List[Document]]


@dataclass
class DialogueSession:
    stuff: Any = field(default_factory=list)