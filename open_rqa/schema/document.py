from dataclasses import dataclass, field

@dataclass
class Document:
    id: str
    text: str
    title: str
    url: str
    source: str
    metadata: dict = field(default_factory=dict)