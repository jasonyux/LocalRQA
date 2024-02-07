from abc import ABC, abstractmethod
from local_rqa.retrievers.base import RetrievalOutput
from local_rqa.schema.document import Document
from local_rqa.schema.dialogue import DialogueSession
from local_rqa.pipelines.retrieval_qa import SimpleRQA
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
    def retrieve(self, question: str) -> RetrievalOutput:
        raise NotImplementedError
    
    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def generate_stream_from_api(self):
        """either this method is used for generate_stream, or model.generate will be used
        so, if you are NOT using acceleration frameworks and is loading models locally, just implement get_model and get_tokenizer
        otherwise, implement this method and return a generator

        Raises:
            NotImplementedError: _description_
        """
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

        if not self.rqa.qa_llm.is_api_model:
            tokenizer = self.rqa.qa_llm.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        return

    def rephrase_question_for_retrieval(self, question: str, history: list) -> str:
        dialogue_session = DialogueSession.from_list(history)
        rephrased_question = self.rqa.rephrase_questions(
            batch_questions=[question],
            batch_dialogue_session=[dialogue_session],
        )[0]
        return rephrased_question

    def retrieve(self, question: str) -> RetrievalOutput:
        output = self.rqa.retriever.retrieve(
            batch_questions=[question]
        )
        return output

    def prepare_prompt_for_generation(self, question: str, retrieved_docs: List[dict], history: List):
        dialogue_session = DialogueSession.from_list(history)
        docs = [Document.from_dict(doc) for doc in retrieved_docs]
        prompt = self.rqa.qa_llm._prepare_question_w_docs(
            question=question,
            docs=docs,
            chat_history_str=dialogue_session.to_string(),
        )
        return prompt

    def get_model(self):
        if self.rqa.qa_llm.is_api_model:
            return None
        return self.rqa.qa_llm.model

    def get_tokenizer(self):
        if self.rqa.qa_llm.is_api_model:
            return None
        return self.rqa.qa_llm.tokenizer

    def generate_stream_from_api(self, input_text: str, **kwargs):
        return self.rqa.qa_llm._generate_stream(input_text, **kwargs)

    @classmethod
    def from_scratch(cls, *args, **kwargs):
        rqa = SimpleRQA.from_scratch(*args, **kwargs)
        return GradioSimpleRQA(rqa)