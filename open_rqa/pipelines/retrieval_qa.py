from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from open_rqa.schema.dialogue import DialogueSession, RQAOutput
from open_rqa.guardrails.base import BaseAnswerGuardrail
from open_rqa.retrievers.base import BaseRetriever
from open_rqa.qa_llms.base import BaseQAModel
from open_rqa.qa_llms.huggingface import HuggingFaceQAModel, HuggingFaceFiDQAModel
from open_rqa.qa_llms.openai import OpenAIQAModel
from open_rqa.guardrails.base import NoopAnswerGuardrail
from open_rqa.pipelines.base import RQAPipeline
from open_rqa.pipelines.prompts import REPHRASE_QUESTION_PROMPT
import logging


logger = logging.getLogger(__name__)


class BaseRQA(RQAPipeline):
    """takes in EXACTLY three components: retriever, qa_llm, answer_guardrail, and runs them in sequence"""

    def __init__(
        self,
        retriever: BaseRetriever,
        qa_llm: BaseQAModel,
        answer_guardrail: BaseAnswerGuardrail,
    ):
        self.components = [retriever, qa_llm, answer_guardrail]
        return


class SimpleRQA(BaseRQA):
    def __init__(
        self,
        retriever: BaseRetriever,
        qa_llm: BaseQAModel,
        answer_guardrail: BaseAnswerGuardrail,
        verbose=False,
    ):
        super().__init__(retriever, qa_llm, answer_guardrail)
        self.qa_llm = qa_llm
        self.verbose = verbose

        # for answering questions
        self._default_tokenization_kwargs = {}
        self._default_generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
        }
        # for rephrasing questions
        self._default_rephrase_kwargs = {
            "max_new_tokens": 128,
        }
        return

    def _batch_generate(self, input_prompts, **generate_kwargs):
        # merge default kwargs with user kwargs
        _gen_kwargs = {**self._default_generate_kwargs, **generate_kwargs}

        gen_output = self.qa_llm.generate(
            batched_prompts=input_prompts,
            tokenization_kwargs=self._default_tokenization_kwargs,
            generation_kwargs=_gen_kwargs,
        )
        responses = gen_output.batch_answers
        return responses

    def rephrase_questions(
        self,
        batch_questions: List[str],
        batch_dialogue_session: List[DialogueSession],
    ) -> List[str]:
        """rephrase every question in batch_questions to be a standalone question

        Args:
            batch_questions (List[str]): _description_
            batch_dialogue_session (List[DialogueSession]): _description_

        Returns:
            List[str]: _description_
        """
        rephrased_questions = []
        need_rephrase_qids = []
        need_rephrase_questions = []
        need_rephrase_dialogue_history_strs = []
        for i, question in enumerate(batch_questions):
            dialogue_history = batch_dialogue_session[i]
            if len(dialogue_history.history) == 0:
                rephrased_questions.append(question)
            else:
                dialogue_history_str = dialogue_history.to_string()
                rephrased_questions.append(None)
                need_rephrase_qids.append(i)
                need_rephrase_questions.append(question)
                need_rephrase_dialogue_history_strs.append(dialogue_history_str)

        if len(need_rephrase_questions) > 0:
            rephrased_questions = self._rephrase_questions(
                need_rephrase_questions, need_rephrase_dialogue_history_strs
            )
            for i, qid in enumerate(need_rephrase_qids):
                rephrased_questions[qid] = rephrased_questions[i]
        return rephrased_questions

    def _rephrase_questions(
        self, questions: List[str], chat_history_strs: List[str]
    ) -> List[str]:
        input_prompts = []
        for question, chat_history_str in zip(questions, chat_history_strs):
            prompt_i = REPHRASE_QUESTION_PROMPT.format(
                question=question,
                chat_history_str=chat_history_str,
                eos_token=self.qa_llm.tokenizer.eos_token,
            )
            input_prompts.append(prompt_i)
            if self.verbose:
                logger.info("[__rephrase_questions] Prompt:")
                logger.info(prompt_i)

        rephrased_qs = self._batch_generate(
            input_prompts,
            **self._default_rephrase_kwargs
        )
        if self.verbose:
            logger.info("[__rephrase_questions] Rephrased questions:")
            logger.info("\n".join(rephrased_qs))
        return rephrased_qs

    def qa(
        self,
        batch_questions: List[str],
        batch_dialogue_session: List[DialogueSession],
    ) -> RQAOutput:
        batch_questions = self.rephrase_questions(
            batch_questions, batch_dialogue_session
        )
        retrieval_qa_output = super().qa(
            batch_questions=batch_questions,
            batch_dialogue_session=batch_dialogue_session,
        )
        self.update_dialogue_session(
            batch_questions=batch_questions,
            retrieval_qa_output=retrieval_qa_output,
            batch_dialogue_session=batch_dialogue_session,
        )
        return retrieval_qa_output

    @staticmethod
    def from_huggingface(
        retriever: BaseRetriever,
        qa_model: Optional[AutoModelForCausalLM] = None,
        qa_tokenizer: Optional[AutoTokenizer] = None,
        qa_model_name_or_path: str = "",
        user_prefix: str = "USER",
        assistant_prefix: str = "ASSISTANT",
        verbose: bool = False,
    ):
        """initialize simple RQA given an already initialized retriever model + huggingface-based qa model

        Args:
            retriever (BaseRetriever): _description_
            qa_model (Optional[AutoModelForCausalLM], optional): _description_. Defaults to None.
            qa_tokenizer (Optional[AutoTokenizer], optional): _description_. Defaults to None.
            qa_model_name_or_path (str, optional): _description_. Defaults to "".
            user_prefix (str, optional): _description_. Defaults to "USER".
            assistant_prefix (str, optional): _description_. Defaults to "ASSISTANT".
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        qa_llm = HuggingFaceQAModel(
            model=qa_model,
            tokenizer=qa_tokenizer,
            model_name_or_path=qa_model_name_or_path,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
        )
        answer_guardrail = NoopAnswerGuardrail()

        rqa = SimpleRQA(
            retriever=retriever,
            qa_llm=qa_llm,
            answer_guardrail=answer_guardrail,
            verbose=verbose
        )
        return rqa

    @staticmethod
    def from_huggingface_fid(
        retriever: BaseRetriever,
        qa_model: Optional[AutoModelForCausalLM] = None,
        qa_tokenizer: Optional[AutoTokenizer] = None,
        qa_model_name_or_path: str = "",
        user_prefix: str = "USER",
        assistant_prefix: str = "ASSISTANT",
        verbose: bool = False,
    ):
        """initialize simple RQA given an already initialized retriever model + huggingface-based qa model

        Args:
            retriever (BaseRetriever): _description_
            qa_model (Optional[AutoModelForCausalLM], optional): Fusion-in-Decoder model. Defaults to None.
            qa_tokenizer (Optional[AutoTokenizer], optional): _description_. Defaults to None.
            qa_model_name_or_path (str, optional): _description_. Defaults to "".
            user_prefix (str, optional): _description_. Defaults to "USER".
            assistant_prefix (str, optional): _description_. Defaults to "ASSISTANT".
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        qa_llm = HuggingFaceFiDQAModel(
            model=qa_model,
            tokenizer=qa_tokenizer,
            model_name_or_path=qa_model_name_or_path,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
        )
        answer_guardrail = NoopAnswerGuardrail()

        rqa = SimpleRQA(
            retriever=retriever,
            qa_llm=qa_llm,
            answer_guardrail=answer_guardrail,
            verbose=verbose
        )
        return rqa

    
    @staticmethod
    def from_openai(
        retriever: BaseRetriever,
        qa_model_name: str,
        user_prefix: str = "USER",
        assistant_prefix: str = "ASSISTANT",
        verbose: bool = False,
    ):
        """initialize simple RQA given an already initialized retriever model + openai-based qa model (e.g. gpt-3.5-turbo)

        Args:
            retriever (BaseRetriever): _description_
            qa_model_name (str): _description_
            user_prefix (str, optional): _description_. Defaults to "USER".
            assistant_prefix (str, optional): _description_. Defaults to "ASSISTANT".
            verbose (bool, optional): _description_. Defaults to False.
        
        Returns:
            _type_: _description_
        """
        qa_llm = OpenAIQAModel(
            model_name=qa_model_name,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
        )
        answer_guardrail = NoopAnswerGuardrail()

        rqa = SimpleRQA(
            retriever=retriever,
            qa_llm=qa_llm,
            answer_guardrail=answer_guardrail,
            verbose=verbose
        )
        return rqa


class AutoRQA(BaseRQA):
    def __init__(
        self,
        retriever: str,
        qa_llm: str,
    ):
        raise NotImplementedError
