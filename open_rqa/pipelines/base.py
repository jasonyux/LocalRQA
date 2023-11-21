from typing import List
from better_abc import abstract_attribute
from dataclasses import asdict
from open_rqa.base import Component
from open_rqa.schema.dialogue import DialogueSession, RQAOutput
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
        """returns an answer given a question and a dialogue history

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
            output_dict = asdict(module_output)
            input_dict.update(output_dict)

        return RQAOutput(
            batched_answers=module_output.batched_answers,
            batched_source_documents=module_output.batched_source_documents,
        )

    def run(self, *args, **kwargs):
        return self.qa(*args, **kwargs)
