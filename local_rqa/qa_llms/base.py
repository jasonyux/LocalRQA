from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from local_rqa.base import Component
from local_rqa.schema.dialogue import DialogueSession
from local_rqa.schema.document import Document


@dataclass
class GenerationOutput:
    batch_answers: List[str]


class BaseQAModel(Component):
    """uses LLMs to generate answers given a question and a set of source documents"""

    run_input_keys = [
        "batch_questions",
        "batch_source_documents",
        "batch_dialogue_session",
    ]
    is_api_model = False

    @abstractmethod
    def r_generate(
        self,
        batch_questions: List[str],
        batch_source_documents: List[List[Document]],
        batch_dialogue_session: List[DialogueSession],
        tokenization_kwargs: Optional[dict] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> GenerationOutput:
        """retrieval augemented generation based on the source documents

        Args:
            batch_questions (List[str]): _description_
            batch_source_documents (List[List[Document]]): _description_
            batch_dialogue_session (List[DialogueSession]): _description_
            tokenization_kwargs (Optional[dict], optional): controls tokenization before generation. Defaults to None = use default in generation model
            generation_kwargs (Optional[dict], optional): controls generation. Defaults to None = use default in generation model

        Raises:
            NotImplementedError: _description_

        Returns:
            GenerationOutput: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        batched_prompts: List[str],
        tokenization_kwargs: dict,
        generation_kwargs: dict,
    ) -> GenerationOutput:
        """generic generation method. Used by r_generate but potentially useful for other purposes (e.g. rephrasing questions)

        Args:
            batched_prompts (List[str]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            GenerationOutput: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_question_w_docs(self, question: str, docs: List[Document], chat_history_str: str) -> List[str]:
        """format question, source documents, and chat history into a prompt for generation. Required for GRADIO demo

        Args:
            question (str): _description_
            docs (List[Document]): _description_
            chat_history_str (str): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            List[str]: _description_
        """
        raise NotImplementedError

    def run(self, *args, **kwargs):
        return self.r_generate(*args, **kwargs)
