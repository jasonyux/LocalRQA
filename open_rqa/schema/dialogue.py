from dataclasses import dataclass, field
from typing import List, Dict
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

    @staticmethod
    def from_dict(dialogue_turn_dict: dict):
        """converts a dictionary to a DialogueTurn object

        Args:
            dialogue_turn_dict (_type_): _description_
        """
        source_documents = dialogue_turn_dict['source_documents']
        for i, source_document in enumerate(source_documents):
            if isinstance(source_document, dict):
                source_documents[i] = Document.from_dict(source_document)
        
        dialogue_turn = DialogueTurn(
            speaker=dialogue_turn_dict['speaker'],
            message=dialogue_turn_dict['message'],
            source_documents=dialogue_turn_dict['source_documents']
        )
        return dialogue_turn

    def to_dict(self):
        """converts the DialogueTurn object into a dictionary

        Returns:
            Dict[str, str]: dictionary representing the dialogue turn
        """
        dialogue_turn_dict = {
            'speaker': self.speaker,
            'message': self.message,
            'source_documents': [doc.to_dict() for doc in self.source_documents]
        }
        return dialogue_turn_dict


@dataclass
class DialogueSession:
    """note that this class assumes user speaks first

    Returns:
        _type_: _description_
    """
    user_prefix: str = "USER"
    assistant_prefix: str = "ASSISTANT"
    history: List[DialogueTurn] = field(default_factory=list)

    def to_string(self) -> str:
        """format dialogue history into a string

        Returns:
            str: formatted dialogue history
        """
        # add </s> to the end of system messages
        history = ""
        for turn in self.history:
            if turn.speaker.lower() in ["system", "assistant"]:
                history += f"{turn.to_string()}</s>"
            else:
                history += f"{turn.to_string()} "
        return history

    def add_user_message(self, user_message: str):
        """add user message as DialogueTurn to dialogue history

        Args:
            user_message (str): user message
        """
        dialogue_turn = DialogueTurn(speaker=self.user_prefix, message=user_message)
        self.history.append(dialogue_turn)
        return

    def add_system_message(self, system_message: str, source_documents: List[Document]):
        """add system message as DialogueTurn to dialogue history

        Args:
            system_message (str): system message
        """
        dialogue_turn = DialogueTurn(
            speaker=self.assistant_prefix, message=system_message, source_documents=source_documents
        )
        self.history.append(dialogue_turn)
        return
    
    def to_list(self):
        """converts the DialogueSession object into a list of dictionaries

        Returns:
            List[Dict[str, str]]: list of dictionaries representing the dialogue session
        """
        dialogue_list = []
        for dialogue_turn in self.history:
            dialogue_dict = {
                'speaker': dialogue_turn.speaker,
                'message': dialogue_turn.message,
                'source_documents': dialogue_turn.source_documents
            }
            dialogue_list.append(dialogue_dict)
        return dialogue_list

    @staticmethod
    def from_list(dialogue_list: List[Dict[str, str]]) -> "DialogueSession":
        """converts a list of dictionaries to a DialogueSession object

        Args:
            dialogue_list (List[Dict[str, str]]): _description_

        Returns:
            _type_: _description_
        """
        dialogue_session = DialogueSession()
        for dialogue_turn_dict in dialogue_list:
            dialogue_turn = DialogueTurn.from_dict(dialogue_turn_dict)
            dialogue_session.history.append(dialogue_turn)
        # adjust user and assistant prefixes
        if len(dialogue_session.history) > 0:
            dialogue_session.user_prefix = dialogue_session.history[0].speaker
        if len(dialogue_session.history) > 1:
            dialogue_session.assistant_prefix = dialogue_session.history[1].speaker
        return dialogue_session


@dataclass
class RQAOutput:
    """stores the answers to a user's question, the relevant source documents, and the UPDATED dialogue history
    """
    batch_answers: List[str]
    batch_source_documents: List[List[Document]]
    batch_dialogue_session: List[DialogueSession]
