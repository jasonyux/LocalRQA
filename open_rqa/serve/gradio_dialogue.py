from dataclasses import dataclass, asdict, field
from open_rqa.schema.document import Document
from open_rqa.schema.dialogue import DialogueSession, SeparatorStyle
from copy import deepcopy
import jsonlines
import re


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


@dataclass
class AnnotationHistory:
    def __init__(self, data_file_path: str, empty_session: GradioDialogueSession, data_indices=''):
        self.empty_session = empty_session
        self.parsed_data_indices = self.parse_int_range(data_indices)
        self.all_sessions, self.all_indices = self.load(data_file_path)
        
        # related to how gradio renders the UI
        self._current_idx = -1
        self._labeled_indicies = set()
        self._submitted = False
        return

    def parse_int_range(self, data_indices):
        if data_indices == '':
            return None
        all_possible_indices = set()
        for s in data_indices.split(','):
            for a, b in re.findall(r'(\d+)-?(\d*)', s):
                all_possible_indices.update(range(int(a), int(a)+1 if b=='' else int(b)+1))
        print('[[AnnotationHistory]] parsed allowed_indices:', all_possible_indices)
        return all_possible_indices

    def _data_idx_filter(self, idx: int):
        if self.parsed_data_indices is None:
            # include all
            return True
        
        if idx not in self.parsed_data_indices:
            return False
        return True

    def load(self, data_file_path: str):
        with jsonlines.open(data_file_path) as fread:
            data = list(fread)
        all_formatted_data = {}
        for idx, d in enumerate(data):
            if not self._data_idx_filter(idx):
                continue
            # choose which ones to load
            question = d['question']
            gold_docs = [Document.from_dict(doc) for doc in d['gold_docs']]
            retrieved_docs = [Document.from_dict(doc) for doc in d['retrieved_docs']]
            gold_answer = d['gold_answer']
            generated_answer = d['generated_answer']

            session = self.empty_session.clone()
            session.add_user_message(question)
            session.add_system_message(generated_answer, retrieved_docs)
            session._tmp_data['gold_answer'] = gold_answer
            session._tmp_data['gold_docs'] = gold_docs

            all_formatted_data[idx] = {
                'session': session,
                'annotation': None  # none is also used by gr.Radio to select nothing
            }
        
        all_indices = list(all_formatted_data.keys())
        return all_formatted_data, all_indices

    def get_next_idx(self):
        self._current_idx = (self._current_idx + 1) % len(self.all_indices)
        return self.all_indices[self._current_idx]

    def get_prev_idx(self):
        self._current_idx = (self._current_idx - 1) % len(self.all_indices)
        return self.all_indices[self._current_idx]

    def get_current_idx(self):
        return self.all_indices[self._current_idx]

    def update_label(self, label):
        if label is None:
            return
        idx = self.get_current_idx()
        self.all_sessions[idx]['annotation'] = label
        self._labeled_indicies.add(idx)
        return

    def get_current_label(self):
        idx = self.get_current_idx()
        return self.all_sessions[idx]['annotation']

    def get_num_labeled(self):
        return len(self._labeled_indicies)
    
    def get_num_to_label(self):
        return len(self.all_sessions)

    def is_all_labeled(self):
        return len(self._labeled_indicies) == len(self.all_sessions)

    def to_jsonl(self, metadata: dict):
        all_annotated_data = []
        for _, v in self.all_sessions.items():
            session = v['session']
            annotation = v['annotation']
            sample = {
                'session': session.to_dict(),
                'annotation': annotation,
                'metadata': metadata,
            }
            all_annotated_data.append(sample)
        return all_annotated_data


if __name__ == "__main__":
    print(default_conversation.get_prompt())