from typing import List
from better_abc import abstract_attribute
from dataclasses import fields
from local_rqa.base import Component
from local_rqa.schema.dialogue import DialogueSession, RQAOutput
import logging


logger = logging.getLogger(__name__)


class RQAPipeline(Component):
    """given a question and a dialogue history, return an answer based on the retrieval and QA models"""

    run_input_keys = ["batch_questions", "batch_dialogue_session"]

    @abstract_attribute
    def components(self) -> List[Component]:
        """a list of components that make up the pipeline

        Returns:
            List[Component]: _description_
        """
        raise NotImplementedError

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
        batch_answers = retrieval_qa_output.batch_answers
        source_documents = retrieval_qa_output.batch_source_documents
        for i, dialogue_session in enumerate(batch_dialogue_session):
            question = batch_questions[i]
            answer = batch_answers[i]
            source_docs = source_documents[i]
            dialogue_session.add_user_message(question)
            dialogue_session.add_system_message(answer, source_documents=source_docs)
        return

    def _prepare_input(self, data_dict: dict, keys: List[str]):
        prepared_input_dict = {}
        for key in keys:
            prepared_input_dict[key] = data_dict[key]
        return prepared_input_dict

    def qa(
        self,
        batch_questions: List[str],
        batch_dialogue_session: List[DialogueSession],
    ) -> RQAOutput:
        """returns an answer given a question and a dialogue history. Assume that you can directly pipe through all the components

        Args:
            batch_questions (List[str]): _description_
            batch_dialogue_session (List[DialogueSession]): _description_

        Returns:
            RQAOutput: a dataclass containing the answer and the source documents
        """
        input_dict = {
            "batch_questions": batch_questions,
            "batch_dialogue_session": batch_dialogue_session,
        }
        for module in self.components:
            module: Component
            processed_input_dict = self._prepare_input(
                input_dict, module.run_input_keys
            )
            module_output = module.run(**processed_input_dict)

            # update any existing keys in the input dict with the output dict
            # do not use asdict(), which is recursive!
            output_dict = {f.name: getattr(module_output, f.name) for f in fields(module_output)}
            input_dict.update(output_dict)

        self.update_dialogue_session(
            batch_questions=batch_questions,
            retrieval_qa_output=module_output,
            batch_dialogue_session=batch_dialogue_session,
        )

        return RQAOutput(
            batch_answers=module_output.batch_answers,
            batch_source_documents=module_output.batch_source_documents,
            batch_dialogue_session=batch_dialogue_session,
        )

    def run(self, *args, **kwargs):
        return self.qa(*args, **kwargs)
