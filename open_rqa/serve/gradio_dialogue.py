from dataclasses import dataclass, asdict, field
from open_rqa.schema.dialogue import DialogueSession, SeparatorStyle
from copy import deepcopy


@dataclass
class GradioDialogueSession:
    """A class that keeps all conversation history."""
    _session: DialogueSession
    _tmp_data: dict = field(default_factory=dict)  # placeholder for intermin http results
    skip_next: bool = False

    def get_prompt(self):
        history_str = self._session.to_string()
        return history_str

    def add_user_message(self, message):
        self._session.add_user_message(message)
        return

    def add_system_message(self, message, source_documents):
        self._session.add_system_message(message, source_documents)
        return

    def to_gradio_chatbot(self):
        # gradio stores chat history as a list of ((user_message, bot_message))
        ret = []
        # assume user starts first
        for i, turn in enumerate(self._session.history):
            if i % 2 == 0:
                ret.append([turn.message, None])
            else:
                ret[-1][-1] = turn.message
        return ret

    def clone(self):
        return GradioDialogueSession(
            _session=self._session.clone(),
            _tmp_data=deepcopy(self._tmp_data),
            skip_next=self.skip_next,
        )

    def to_dict(self):
        almost_dict = asdict(self)
        almost_dict['_session'].pop('sep_style', None)  # cannot be serialized
        return almost_dict

conv_vicuna_v1 = GradioDialogueSession(
    _session=DialogueSession(
        user_prefix="USER",
        assistant_prefix="ASSISTANT",
        sep_style=SeparatorStyle.TWO,
        sep_user=" ",
        sep_sys="</s>",
    ),
    skip_next=False,
)

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_vicuna_v1,
    # add your own conversation templates here
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())