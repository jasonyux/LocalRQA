import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from open_rqa.schema.document import Document
from open_rqa.serve.gradio_dialogue import default_conversation, conv_templates, SeparatorStyle, GradioDialogueSession
from open_rqa.constants import SERVER_LOGDIR, QA_MODERATION_MSG, SERVER_ERROR_MSG
from open_rqa.utils import init_logger


logger = init_logger(filename="logs/gradio_web_server.log")


headers = {"User-Agent": "LocalRQA Client"}
args: argparse.Namespace


no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


NUM_DOC_TO_RETRIEVE = 4


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]
    }
    text = text.replace("\n", "")
    json_data = {'input': text}
    try:
        # ret = requests.post(url, headers=headers, data=data, timeout=5)
        ret = requests.post(url, headers=headers, json=json_data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False
    return flagged


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(SERVER_LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    logger.info(f"Models: {models}")
    return models


def document_view(idx: int, document: Document):
    view = gr.Markdown(
        value=document.fmt_content,
        autoscroll=False,
        visible=True,
        header_links=True,
        max_height=50,
        elem_classes=["retr_document"],
    )
    return view


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    state = default_conversation.clone()
    return state


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    state = default_conversation.clone()
    return state


def vote_last_response(state: GradioDialogueSession, vote_type, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": args.model_id,
            "state": state.to_dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", request)
    return ("",) + (disable_btn,) * 3


def regenerate(state: GradioDialogueSession, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")

    prev_system_msg = state._session.history.pop()
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.clone()
    tboxes = []
    for i in range(NUM_DOC_TO_RETRIEVE):
        doc = Document(
            page_content="(empty)",
            fmt_content="(empty)",
            metadata={}
        )
        t = document_view(i, doc)
        tboxes.append(t)
    return (state, state.to_gradio_chatbot(), "") + tuple(tboxes) + (disable_btn,) * 5 + (enable_btn,)


def add_text(state: GradioDialogueSession, text, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 6
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), QA_MODERATION_MSG) + (no_change_btn,) * 6

    text = text[:1536]  # Hard cut-off
    state.add_user_message(text)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 6


def http_retrieve(state: GradioDialogueSession, request: gr.Request):
    logger.info(f"http_retrieve. ip: {request.client.host}")
    if state.skip_next:
        # This generate call is skipped due to invalid inputs (e.g., empty inputs)
        tboxes = []
        for i, doc in enumerate(state._tmp_data.get('retrieved_docs', [])):
            t = document_view(i, doc)
            tboxes.append(t)
        return tuple([state] + tboxes)

    start_tstamp = time.time()
    model_name = args.model_id

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(
        controller_url + "/get_worker_address",
        json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")
    try:
        # Make requests
        _session = state._session.clone()
        _session.history.pop()  # remove the last user message from the history, as its passed in separately
        pload = {
            "model_input": {
                "question": state._session.history[-1].message,
                "history": _session.to_list(),
            },
            "k": 4,
        }
        response = requests.post(
            worker_addr + "/worker_retrieve",
            headers=headers, json=pload, stream=False, timeout=10
        )
        data = response.json()
        documents = data["documents"]
        rephrased_question = data["rephrased_question"]
        fmt_documents = [Document.from_dict(doc) for doc in documents]

        state._tmp_data['retrieved_docs'] = fmt_documents
        state._tmp_data['rephrased_question'] = rephrased_question
        finish_tstamp = time.time()
        state._tmp_data['r_start'] = start_tstamp
        state._tmp_data['r_finish'] = finish_tstamp
    except requests.exceptions.RequestException as _:
        state.skip_next = True
        tboxes = []
        for i in range(NUM_DOC_TO_RETRIEVE):
            doc = Document(
                page_content=SERVER_ERROR_MSG,
                fmt_content=SERVER_ERROR_MSG,
                metadata={}
            )
            t = document_view(i, doc)
            tboxes.append(t)
        return tuple([state] + tboxes)

    tboxes = []
    for i, doc in enumerate(fmt_documents):
        t = document_view(i, doc)
        tboxes.append(t)
    return tuple([state] + tboxes)


def http_generate(state: GradioDialogueSession, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_generate. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = args.model_id

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 6
        return

    if len(state._session.history) == 0:
        # First round of conversation
        template_name = args.conv_template
        new_state = conv_templates[template_name].clone()
        new_state.add_user_message(state._session.history[-2].message)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(
        controller_url + "/get_worker_address",
        json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.add_system_message(SERVER_ERROR_MSG, [])
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Make requests
    retrieved_docs = state._tmp_data.get('retrieved_docs', [])
    _session = state._session.clone()
    _session.history.pop()  # remove the last user message from the history, as its passed in separately
    pload = {
        "model": model_name,
        "model_input": {
            "retrieved_docs": [doc.to_dict() for doc in retrieved_docs],
            "question": state._session.history[-1].message,
            "history": _session.to_list(),
        },
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state._session.sep_user if state._session.sep_style in [SeparatorStyle.SINGLE] else state._session.sep_sys,
    }
    logger.info(f"==== request ====\n{pload}")

    state.add_system_message("▌", retrieved_docs)
    
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 6

    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"].strip()
                    state._session.history[-1].message = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 6
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state._session.history[-1].message = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as _:
        state._session.history[-1].message = SERVER_ERROR_MSG
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
        return

    state._session.history[-1].message = state._session.history[-1].message[:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 6

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "r_start": state._tmp_data['r_start'],
            "r_finish": state._tmp_data['r_finish'],
            "state": state.to_dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# LocalRQA
[[Code](https://github.com/jasonyux/LocalRQA)] | 📚 [[LocalRQA](https://arxiv.org/abs/xxxxx)]
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

.retr_document {
    max-height: 175px;
    min-height: 175px;
}

""".strip()


block_js = """
""".strip()


def build_demo(embed_mode):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LocalRQA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)
        
        with gr.Column():
            ### retrieval part
            retrieved_docs = []
            with gr.Row(elem_id="retrieval"):
                for i in range(NUM_DOC_TO_RETRIEVE):
                    with gr.Tab(f"Retrieved document {i+1}"):
                        doc = Document(
                            page_content="(empty)",
                            fmt_content="(empty)",
                            metadata={}
                        )
                        t = document_view(i, doc)
                        retrieved_docs.append(t)
            
            chatbot = gr.Chatbot(elem_id="chatbot", label="LocalRQA Chatbot", height=550)

            ## example and gen params
            with gr.Row(equal_height=True):
                with gr.Column(scale=5):
                    gr.Examples(
                        examples=[[ex] for ex in args.example] or [['(use --example to specify example questions here)']], 
                        inputs=[textbox]
                    )
                with gr.Column(scale=5):
                    with gr.Accordion("Parameters", open=False):
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.1, interactive=True, label="Temperature",)
                        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.1, interactive=True, label="Top P",)
                        max_output_tokens = gr.Slider(minimum=0, maximum=512, value=350, step=16, interactive=True, label="Max output tokens",)

            ## user input
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(value="Send", variant="primary")
            
            ## buttons
            with gr.Row(elem_id="buttons") as _:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn, submit_btn]
        upvote_btn.click(
            upvote_last_response,
            [state],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        downvote_btn.click(
            downvote_last_response,
            [state],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        flag_btn.click(
            flag_last_response,
            [state],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )

        regenerate_btn.click(
            regenerate,
            [state],
            [state, chatbot, textbox] + btn_list,
            queue=False
        ).then(
            http_retrieve,
            [state],
            [state] + retrieved_docs
        ).then(
            http_generate,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox] + retrieved_docs + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox],
            [state, chatbot, textbox] + btn_list,
            queue=False
        ).then(
            http_retrieve,
            [state],
            [state] + retrieved_docs
        ).then(
            http_generate,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        submit_btn.click(
            add_text,
            [state, textbox],
            [state, chatbot, textbox] + btn_list,
            queue=False
        ).then(
            http_retrieve,
            [state],
            [state] + retrieved_docs
        ).then(
            http_generate,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )
        

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state],
                _js=block_js,
                queue=False
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--model_id", type=str, default="simple_rqa")
    parser.add_argument('--example', action='append', help='Example question displayed on demo page')
    parser.add_argument("--conv_template", type=str, default="default")  # see conv_templates in gradio_dialogue.py
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()
    assert args.model_id in models, f"Model {args.model_id} not found in {models}. Are you sure you have started the worker with {args.model_id}?"

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(
        concurrency_count=args.concurrency_count,
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )