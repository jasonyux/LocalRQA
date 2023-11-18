from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.schema.document import Document


@dataclass
class GenerationOutput:
    batched_answers: List[str]


class BaseQAModel(ABC):
    """uses LLMs to generate answers given a question and a set of source documents
    """
    @abstractmethod
    def r_generate(self,
        batch_questions: List[str],
        batch_source_documents: List[List[Document]],
        batch_dialogue_history: List[DialogueSession],
    ) -> GenerationOutput:
        """conditional generation based on the source documents

        Args:
            batch_questions (List[str]): _description_
            batch_source_documents (List[List[Document]]): _description_
            batch_dialogue_history (List[DialogueSession]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            GenerationOutput: _description_
        """
        raise NotImplementedError

