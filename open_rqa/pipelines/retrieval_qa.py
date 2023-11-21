from typing import List
from open_rqa.schema.dialogue import DialogueSession, RQAOutput
from open_rqa.guardrails.base import BaseAnswerGuardrail
from open_rqa.retrievers.base import BaseRetriever
from open_rqa.qa_llms.base import BaseQAModel
from open_rqa.pipelines.base import RQAPipeline
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


class AutoRQA(BaseRQA):
    def __init__(
        self,
        retriever: str,
        qa_llm: str,
    ):
        raise NotImplementedError


class SimpleRQA(BaseRQA):
    REPHRASE_QUESTION_PROMPT = """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History: {chat_history_str}
    Follow Up Input: {question}{eos_token}
    Standalone question:
    """.replace(
        " "*4, ""
    ).strip()
    
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

        # TODO: with TGI we have quite a different score than not using it. perhaps check if it was doing quantization automatically
        self.__default_generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "num_beams": 1,
            # "repetition_penalty": 1.00,  # cause CUDA-assertion when used with TGI
            # "typical_p": 0.999,  # cause CUDA-assertion when used with TGI
            "eos_token_id": None if self.qa_llm.is_api_model else self.qa_llm.tokenizer.eos_token_id, 
            "early_stopping": True,
        }
        return

    def _batch_generate(self, input_prompts, **generate_kwargs):
        # merge default kwargs with user kwargs
        _gen_kwargs = {**self.__default_generate_kwargs, **generate_kwargs}

        gen_output = self.qa_llm.generate(
            batched_prompts=input_prompts,
            tokenization_kwargs={},
            generation_kwargs=_gen_kwargs,
        )
        responses = gen_output.batch_answers
        responses =[r.strip().replace(self.qa_llm.tokenizer.eos_token, "") for r in responses]
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
            prompt_i = self.REPHRASE_QUESTION_PROMPT.format(
                question=question,
                chat_history_str=chat_history_str,
                eos_token=self.qa_llm.tokenizer.eos_token,
            )
            input_prompts.append(prompt_i)
            if self.verbose:
                logger.info("[__rephrase_questions] Prompt:")
                logger.info(prompt_i)

        rephrased_qs = self._batch_generate(input_prompts, max_new_tokens=128)
        if self.verbose:
            logger.info("[__rephrase_questions] Rephrased questions:")
            logger.info("\n".join(rephrased_qs))
        return rephrased_qs

    def update_dialogue_session(
        self,
        batch_questions: List[str],
        retrieval_qa_output: RQAOutput,
        batch_dialogue_session: List[DialogueSession],
    ):
        """update the dialogue session with the question from the current turn and the answer from the system

        Args:
            batch_questions (List[str]): _description_
            retrieval_qa_output (RQAOutput): _description_
            batch_dialogue_session (List[DialogueSession]): _description_
        """
        batch_answers = retrieval_qa_output.batched_answers
        for i, dialogue_session in enumerate(batch_dialogue_session):
            question = batch_questions[i]
            answer = batch_answers[i]
            dialogue_session.add_user_message(question)
            dialogue_session.add_system_message(answer)
        return

    def qa(
        self,
        batch_questions: List[str],
        batch_dialogue_session: List[DialogueSession],
    ) -> List[RQAOutput]:
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
