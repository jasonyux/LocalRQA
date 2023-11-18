from abc import ABC, abstractmethod
from typing import List
from open_rqa.schema.document import Document
from open_rqa.schema.dialogue import DialogueSession, RQAOutput


class BaseAnswerGuardrail(ABC):
    """performs actions such as fact checking, safety filtering, etc,
    """
    @abstractmethod
    def guardrail(self,
        batch_questions: List[str],
        batch_source_documents: List[List[Document]],
        batch_dialogue_history: List[DialogueSession],
        batch_generated_answers: List[str]
    ) -> RQAOutput:
        """post-processing the response before returning to the user

        Args:
            batch_questions (List[str]): _description_
            batch_source_documents (List[List[Document]]): _description_
            batch_dialogue_history (List[DialogueSession]): _description_
            batch_generated_answers (List[str]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            RQAOutput: _description_
        """
        raise NotImplementedError


class NoopAnswerGuardrail(BaseAnswerGuardrail):
    """dummy answer guardrail that passes the answer through
    """
    def guardrail(self,
        batch_questions: List[str],
        batch_source_documents: List[List[Document]],
        batch_dialogue_history: List[DialogueSession],
        batch_generated_answers: List[str]
    ) -> RQAOutput:
        return RQAOutput(
            batched_answers=batch_generated_answers,
            batched_source_documents=batch_source_documents,
        )