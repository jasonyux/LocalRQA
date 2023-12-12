from dataclasses import dataclass, field
from typing import List
from open_rqa.schema.document import Document


@dataclass
class DialogueTurn:
    """stores a single dialogue turn. Added source_documents which MAY be useful later
    """
    speaker: str
    message: str
    source_documents: List[Document] = field(default_factory=list)

    def to_string(self) -> str:
        """converts the dialogue turn into a string

        Returns:
            str: string representation of the dialogue turn
        """
        return f"{self.speaker}: {self.message}"


@dataclass
class DialogueSession:
    history: List[DialogueTurn] = field(default_factory=list)

    def to_string(self) -> str:
        """format dialogue history into a string

        Returns:
            str: formatted dialogue history
        """
        return "\n".join([turn.to_string() for turn in self.history])

    def add_user_message(self, user_message: str):
        """add user message as DialogueTurn to dialogue history

        Args:
            user_message (str): user message
        """
        dialogue_turn = DialogueTurn(speaker="user", message=user_message)
        self.history.append(dialogue_turn)
        return

    def add_system_message(self, system_message: str, source_documents: List[Document]):
        """add system message as DialogueTurn to dialogue history

        Args:
            system_message (str): system message
        """
        dialogue_turn = DialogueTurn(
            speaker="system", message=system_message, source_documents=source_documents
        )
        self.history.append(dialogue_turn)
        return


@dataclass
class RQAOutput:
    """stores the answers to a user's question, the relevant source documents, and the UPDATED dialogue history
    """
    batch_answers: List[str]
    batch_source_documents: List[List[Document]]
    batch_dialogue_session: List[DialogueSession]
