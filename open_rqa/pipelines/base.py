from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from open_rqa.schema.dialogue import DialogueSession, RQAOutput
from open_rqa.guardrails.base import BaseAnswerGuardrail
from open_rqa.retrievers.base import BaseRetriever
from open_rqa.qa_llms.base import BaseQAModel
import logging


logger = logging.getLogger(__name__)


class RetrievalQAPipeline(ABC):
    """given a question and a dialogue history, return an answer based on the retrieval and QA models
    """
    @abstractmethod
    def qa(self,
        batch_questions: List[str],
        batch_dialogue_history: List[DialogueSession],
    ) -> RQAOutput:
        raise NotImplementedError


class RetrievalQA(RetrievalQAPipeline):
    def __init__(self,
        retriever: BaseRetriever,
        qa_llm: BaseQAModel,
        answer_guardrail: BaseAnswerGuardrail
    ):
        self.retriever = retriever
        self.qa_llm = qa_llm
        self.guardrail = answer_guardrail
        return

    def qa(self, batch_questions: List[str], batch_dialogue_history: List[DialogueSession]) -> RQAOutput:
        # retrieve relevant documents
        retrieval_output = self.retriever.retrieve(batch_questions, batch_dialogue_history)
        # generate answers
        raw_gen_output = self.qa_llm.r_generate(retrieval_output)
        # guardrail
        gen_output = self.guardrail.guardrail(
            batch_questions=batch_questions,
            batch_source_documents=retrieval_output.source_documents,
            batch_dialogue_history=batch_dialogue_history,
            batch_generated_answers=raw_gen_output.batched_answers
        )
        return gen_output