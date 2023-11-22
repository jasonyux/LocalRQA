from dataclasses import dataclass, field
from typing import Any, List
from open_rqa.schema.document import Document


@dataclass
class DialogueSession:
    history: Any = field(default_factory=list)

    def to_string(self) -> str:
        """format dialogue history into a string

        Returns:
            str: formatted dialogue history
        """
        return "\n".join([str(h) for h in self.history])

    def add_user_message(self, user_message: str):
        """add user message to dialogue history

        Args:
            user_message (str): user message
        """
        raise NotImplementedError

    def add_system_message(self, system_message: str):
        """add system message to dialogue history

        Args:
            system_message (str): system message
        """
        raise NotImplementedError


@dataclass
class RQAOutput:
    """stores the answers to a user's question, the relevant source documents, and the UPDATED dialogue history
    """
    batch_answers: List[str]
    batch_source_documents: List[List[Document]]
    batch_dialogue_session: List[DialogueSession]
