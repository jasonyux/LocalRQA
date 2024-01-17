from abc import ABC, abstractmethod
from open_rqa.retrievers.base import RetrievalOutput
from open_rqa.schema.document import Document
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.pipelines.retrieval_qa import SimpleRQA
from typing import List
import logging


logger = logging.getLogger(__name__)


class GradioRQA(ABC):
    """key methods used in gradio

    Args:
        ABC (_type_): _description_

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
    """
    def rephrase_question_for_retrieval(self, question: str, history: list) -> str:
        return question  # noop
    
    @abstractmethod
    def retrieve(self) -> RetrievalOutput:
        raise NotImplementedError
    
    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def prepare_prompt_for_generation(self, question: str, retrieved_docs: List[dict], history: List):
        raise NotImplementedError


class GradioSimpleRQA(GradioRQA):
    """wrapper class for SimpleRQA used in gradio

    Args:
        GradioRQA (_type_): _description_
    """
    def __init__(self, rqa: SimpleRQA):
        self.rqa = rqa
        return

    def rephrase_question_for_retrieval(self, question: str, history: list) -> str:
        dialogue_session = DialogueSession.from_list(history)
        rephrased_question = self.rqa.rephrase_questions(
            batch_questions=[question],
            batch_dialogue_session=[dialogue_session],
        )[0]
        return rephrased_question

    def retrieve(self) -> RetrievalOutput:
        return RetrievalOutput(
            batch_source_documents = [[
                Document(page_content='text a', fmt_content='DBFS is databricks file system.', metadata={}),
                Document(page_content='text b', fmt_content='Databricks is a moving company that works on moving bricks.', metadata={}),
            ]]
        )

    def get_model(self):
        return self.rqa.qa_llm.model

    def get_tokenizer(self):
        return self.rqa.qa_llm.tokenizer

    def prepare_prompt_for_generation(self, question: str, retrieved_docs: List[dict], history: List):
        dialogue_session = DialogueSession.from_list(history)
        docs = [Document.from_dict(doc) for doc in retrieved_docs]
        prompt = self.rqa.qa_llm._prepare_question_w_docs(
            question=question,
            docs=docs,
            chat_history_str=dialogue_session.to_string(),
        )
        return prompt

    @classmethod
    def from_scratch(cls, *args, **kwargs):
        rqa = SimpleRQA.from_scratch(*args, **kwargs)
        return GradioSimpleRQA(rqa)