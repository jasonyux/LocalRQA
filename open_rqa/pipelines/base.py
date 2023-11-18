from typing import List
from abc import ABC, abstractmethod
from open_rqa.schema.dialogue import DialogueSession, RQAOutput
import logging


logger = logging.getLogger(__name__)


class RQAPipeline(ABC):
    """given a question and a dialogue history, return an answer based on the retrieval and QA models
    """
    @abstractmethod
    def qa(self,
        batch_questions: List[str],
        batch_dialogue_history: List[DialogueSession],
    ) -> RQAOutput:
        raise NotImplementedError