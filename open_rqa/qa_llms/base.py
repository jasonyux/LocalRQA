from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from open_rqa.base import Component
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.schema.document import Document


@dataclass
class GenerationOutput:
    batch_answers: List[str]


class BaseQAModel(Component):
    """uses LLMs to generate answers given a question and a set of source documents"""

    run_input_keys = [
        "batch_questions",
        "batch_source_documents",
        "batch_dialogue_history",
    ]

    @abstractmethod
    def r_generate(
        self,
        batch_questions: List[str],
        batch_source_documents: List[List[Document]],
        batch_dialogue_history: List[DialogueSession],
        tokenization_kwargs: Optional[dict] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> GenerationOutput:
        """retrieval augemented generation based on the source documents

        Args:
            batch_questions (List[str]): _description_
            batch_source_documents (List[List[Document]]): _description_
            batch_dialogue_history (List[DialogueSession]): _description_
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

    def run(self, *args, **kwargs):
        return self.r_generate(*args, **kwargs)
