from abc import abstractmethod
from typing import List
from local_rqa.schema.document import Document
from local_rqa.schema.dialogue import DialogueSession, RQAOutput
from local_rqa.base import Component


class BaseAnswerGuardrail(Component):
    """performs actions such as fact checking, safety filtering, etc,"""
    run_input_keys = [
        "batch_questions",
        "batch_source_documents",
        "batch_dialogue_session",
        "batch_answers",
    ]

    @abstractmethod
    def guardrail(
        self,
        batch_questions: List[str],
        batch_source_documents: List[List[Document]],
        batch_dialogue_session: List[DialogueSession],
        batch_answers: List[str],
    ) -> RQAOutput:
        """post-processing the response before returning to the user

        Args:
            batch_questions (List[str]): _description_
            batch_source_documents (List[List[Document]]): _description_
            batch_dialogue_session (List[DialogueSession]): _description_
            batch_answers (List[str]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            RQAOutput: _description_
        """
        raise NotImplementedError

    def run(self, *args, **kwargs):
        return self.guardrail(*args, **kwargs)


class NoopAnswerGuardrail(BaseAnswerGuardrail):
    """dummy answer guardrail that passes the answer through"""

    def guardrail(
        self,
        batch_questions: List[str],
        batch_source_documents: List[List[Document]],
        batch_dialogue_session: List[DialogueSession],
        batch_answers: List[str],
    ) -> RQAOutput:
        return RQAOutput(
            batch_answers=batch_answers,
            batch_source_documents=batch_source_documents,
            batch_dialogue_session=batch_dialogue_session,
        )
