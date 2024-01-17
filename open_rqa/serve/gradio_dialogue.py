from dataclasses import dataclass, asdict, field
from open_rqa.schema.dialogue import DialogueSession, SeparatorStyle


@dataclass
class GradioDialogueSession:
    """A class that keeps all conversation history."""
    _session: DialogueSession
    _tmp_data: dict = field(default_factory=dict)  # placeholder for intermin http results
    skip_next: bool = False

    def get_prompt(self):
        # messages = self.messages
        # if len(messages) > 0 and type(messages[0][1]) is tuple:
        #     messages = self.messages.copy()
        #     init_role, init_msg = messages[0].copy()
        #     init_msg = init_msg[0].replace("<image>", "").strip()
        #     if 'mmtag' in self.version:
        #         messages[0] = (init_role, init_msg)
        #         messages.insert(0, (self.roles[0], "<Image><image></Image>"))
        #         messages.insert(1, (self.roles[1], "Received."))
        #     else:
        #         messages[0] = (init_role, "<image>\n" + init_msg)

        # if self.sep_style == SeparatorStyle.SINGLE:
        #     ret = self.system + self.sep
        #     for role, message in messages:
        #         if message:
        #             if type(message) is tuple:
        #                 message, _, _ = message
        #             ret += role + ": " + message + self.sep
        #         else:
        #             ret += role + ":"
        # elif self.sep_style == SeparatorStyle.TWO:
        #     seps = [self.sep, self.sep2]
        #     ret = self.system + seps[0]
        #     for i, (role, message) in enumerate(messages):
        #         if message:
        #             if type(message) is tuple:
        #                 message, _, _ = message
        #             ret += role + ": " + message + seps[i % 2]
        #         else:
        #             ret += role + ":"
        # elif self.sep_style == SeparatorStyle.MPT:
        #     ret = self.system + self.sep
        #     for role, message in messages:
        #         if message:
        #             if type(message) is tuple:
        #                 message, _, _ = message
        #             ret += role + message + self.sep
        #         else:
        #             ret += role
        # elif self.sep_style == SeparatorStyle.LLAMA_2:
        #     wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
        #     wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
        #     ret = ""

        #     for i, (role, message) in enumerate(messages):
        #         if i == 0:
        #             assert message, "first message should not be none"
        #             assert role == self.roles[0], "first message should come from user"
        #         if message:
        #             if type(message) is tuple:
        #                 message, _, _ = message
        #             if i == 0: message = wrap_sys(self.system) + message
        #             if i % 2 == 0:
        #                 message = wrap_inst(message)
        #                 ret += self.sep + message
        #             else:
        #                 ret += " " + message + " " + self.sep2
        #         else:
        #             ret += ""
        #     ret = ret.lstrip(self.sep)
        # elif self.sep_style == SeparatorStyle.PLAIN:
        #     seps = [self.sep, self.sep2]
        #     ret = self.system
        #     for i, (role, message) in enumerate(messages):
        #         if message:
        #             if type(message) is tuple:
        #                 message, _, _ = message
        #             ret += message + seps[i % 2]
        #         else:
        #             ret += ""
        # else:
        #     raise ValueError(f"Invalid style: {self.sep_style}")
        history_str = self._session.to_string()
        return history_str

    def add_user_message(self, message):
        self._session.add_user_message(message)
        return

    def add_system_message(self, message, source_documents):
        self._session.add_system_message(message, source_documents)
        return

    def get_images(self, return_pil=False):
        images = []
        return images

    def to_gradio_chatbot(self):
        # gradio stores chat history as a list of ((user_message, bot_message))
        ret = []
        # assume user starts first
        for i, turn in enumerate(self._session.history):
            if i % 2 == 0:
                ret.append([turn.message, None])
            else:
                ret[-1][-1] = turn.message

        # for i, (role, msg) in enumerate(self.messages[self.offset:]):
        #     if i % 2 == 0:
        #         if type(msg) is tuple:
        #             import base64
        #             from io import BytesIO
        #             msg, image, image_process_mode = msg
        #             max_hw, min_hw = max(image.size), min(image.size)
        #             aspect_ratio = max_hw / min_hw
        #             max_len, min_len = 800, 400
        #             shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        #             longest_edge = int(shortest_edge * aspect_ratio)
        #             W, H = image.size
        #             if H > W:
        #                 H, W = longest_edge, shortest_edge
        #             else:
        #                 H, W = shortest_edge, longest_edge
        #             image = image.resize((W, H))
        #             buffered = BytesIO()
        #             image.save(buffered, format="JPEG")
        #             img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        #             img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
        #             msg = img_str + msg.replace('<image>', '').strip()
        #             ret.append([msg, None])
        #         else:
        #             ret.append([msg, None])
        #     else:
        #         ret[-1][-1] = msg
        return ret

    def clone(self):
        return GradioDialogueSession(
            _session=self._session.clone(),
            skip_next=self.skip_next,
        )
        # return OldConversation(
        #     system=self.system,
        #     roles=self.roles,
        #     messages=[[x, y] for x, y in self.messages],
        #     offset=self.offset,
        #     sep_style=self.sep_style,
        #     sep=self.sep,
        #     sep2=self.sep2,
        #     version=self.version
        # )

    def to_dict(self):
        almost_dict = asdict(self)
        almost_dict['_session'].pop('sep_style', None)  # cannot be serialized
        return almost_dict
        # if len(self.get_images()) > 0:
        #     return {
        #         "system": self.system,
        #         "roles": self.roles,
        #         "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
        #         "offset": self.offset,
        #         "sep": self.sep,
        #         "sep2": self.sep2,
        #     }
        # return {
        #     "system": self.system,
        #     "roles": self.roles,
        #     "messages": self.messages,
        #     "offset": self.offset,
        #     "sep": self.sep,
        #     "sep2": self.sep2,
        # }

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
# conv_vicuna_v1 = OldConversation(
#     system="A chat between a curious user and an artificial intelligence assistant. "
#     "The assistant gives helpful, detailed, and polite answers to the user's questions.",
#     roles=("USER", "ASSISTANT"),
#     messages=(),
#     sep_style=SeparatorStyle.TWO,
#     sep=" ",
#     sep2="</s>",
# )

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())