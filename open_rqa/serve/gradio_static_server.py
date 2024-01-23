import argparse
import datetime
import jsonlines
import os
import gradio as gr

from open_rqa.evaluation.metrics import is_almost_same_document
from open_rqa.schema.document import Document
from open_rqa.serve.gradio_dialogue import default_conversation, GradioDialogueSession, AnnotationHistory
from open_rqa.constants import SERVER_LOGDIR
from open_rqa.utils import init_logger


logger = init_logger(filename="logs/gradio_static_server.log")


headers = {"User-Agent": "LocalRQA Client"}
args: argparse.Namespace


no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


NUM_DOC_TO_RETRIEVE = 2 + 1  # +1 for the gold document
ANN_CORRECT = "👍 correct"
ANN_INCORRECT = "👎 incorrect"


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(SERVER_LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-{t.hour:02d}-{t.minute:02d}-annotations.jsonl")
    return name


def document_view(idx: int, document: Document):
    view = gr.Markdown(
        value=document.fmt_content,
        visible=True,
        elem_classes=["retr_document"],
        line_breaks=True,
    )
    return view


def render_single_session(session: GradioDialogueSession):
    chatbot_content = session.to_gradio_chatbot()
    retr_documents = session._session.history[-1].source_documents
    gold_docs = session._tmp_data['gold_docs']
    # gold_answer = session._tmp_data['gold_answer']

    ### only use unique documents from the gold + retrieved documents set
    all_docs_to_display = [gold_docs[0]]
    for doc in gold_docs[1:] + retr_documents:
        if not any(is_almost_same_document(doc, d) for d in all_docs_to_display):
            all_docs_to_display.append(doc)
    all_docs_to_display = all_docs_to_display[:NUM_DOC_TO_RETRIEVE]
    if len(all_docs_to_display) < NUM_DOC_TO_RETRIEVE:
        # pad with empty documents
        all_docs_to_display += [Document(
            page_content="(empty)",
            fmt_content="(empty)",
            metadata={}
        )] * (NUM_DOC_TO_RETRIEVE - len(all_docs_to_display))

    tboxes = []
    for i, doc in enumerate(all_docs_to_display):
        t = document_view(i, doc)
        tboxes.append(t)
    return chatbot_content, tboxes


def render_next_session(state: AnnotationHistory, submit_btn):
    idx_to_render = state.get_next_idx()
    chatbot, tboxes = render_single_session(state.all_sessions[idx_to_render]['session'])

    if state.is_all_labeled():
        # submit_btn.interactive = True
        submit_btn = enable_btn
    if state._submitted:
        submit_btn = disable_btn

    radio_label = state.get_current_label()
    return tuple([state, chatbot] + tboxes + [radio_label, submit_btn])


def render_prev_session(state: AnnotationHistory, submit_btn):
    idx_to_render = state.get_prev_idx()
    chatbot, tboxes = render_single_session(state.all_sessions[idx_to_render]['session'])

    if state.is_all_labeled():
        # submit_btn.interactive = True
        submit_btn = enable_btn
    if state._submitted:
        submit_btn = disable_btn
    
    radio_label = state.get_current_label()
    return tuple([state, chatbot] + tboxes + [radio_label, submit_btn])


def vote_response(state: AnnotationHistory, radio_choice, submit_btn):
    state.update_label(radio_choice)

    if state.is_all_labeled():
        # submit_btn.interactive = True
        submit_btn = enable_btn
    if state._submitted:
        submit_btn = disable_btn

    pbar = gr.Markdown(f"Done: {state.get_num_labeled()} / {state.get_num_to_label()}")
    return state, pbar, submit_btn


def load_demo(url_params, chatbot, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    state = AnnotationHistory(
        data_file_path=args.file_path,
        empty_session=default_conversation.clone(),
        data_indices=args.include_idx,
    )
    idx_to_render = state.get_next_idx()
    chatbot, tboxes = render_single_session(state.all_sessions[idx_to_render]['session'])

    pbar = gr.Markdown(f"Done: 0 / {state.get_num_to_label()}")
    return tuple([state, chatbot, pbar] + tboxes)


def save_annotations(state: AnnotationHistory):
    all_annotations = state.to_jsonl(metadata={'file': args.file_path})
    try:
        with jsonlines.open(get_conv_log_filename(), "w") as fwrite:
            fwrite.write_all(all_annotations)
    except Exception as e:
        logger.error(f"Failed to save annotations: {e}")
        raise gr.Error("Failed to save annotations. Please try again later or refresh the page.")

    state._submitted = True
    gr.Info('Submission successful! Thank you for your participation!')
    pbar = gr.Markdown(f"Done: {state.get_num_labeled()} / {state.get_num_to_label()}. Submission successful!")
    return disable_btn, pbar


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
    max-height: 300px;
    min-height: 300px;
}

""".strip()


block_js = """
""".strip()


def build_demo(embed_mode):
    with gr.Blocks(title="LocalRQA", theme=gr.themes.Default(), css=block_css) as demo:
        # dummy variable since all components here need to be gr Componetns
        # the real ones are initialized inside the demo.load() function
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)
        
        with gr.Column():
            ### retrieval part
            reference_docs = []
            with gr.Row(elem_id="retrieval"):
                for i in range(NUM_DOC_TO_RETRIEVE):
                    with gr.Tab(f"Reference document {i+1}"):
                        doc = Document(
                            page_content="(empty)",
                            fmt_content="(empty)",
                            metadata={}
                        )
                        t = document_view(i, doc)
                    reference_docs.append(t)
            
            chatbot = gr.Chatbot(elem_id="chatbot", label="LocalRQA Chatbot", height=450)
            
            ## buttons
            radio = gr.Radio(
                [ANN_CORRECT, ANN_INCORRECT],
                label="Correctness",
                info="A response is correct if it answered the user's question AND contains the correct information from the reference documents.",
            )
            with gr.Row(elem_id="buttons") as _:
                previous = gr.Button(value="⬅️  Previous", interactive=True)
                next_btn = gr.Button(value="Next  ➡️", interactive=True)
                submit_btn = gr.Button(value="Done", variant="primary", interactive=False)

            pbar = gr.Markdown("Done: 0 / 0")
        
        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        next_btn.click(
            render_next_session,
            [state] + [submit_btn],
            [state, chatbot] + reference_docs + [radio, submit_btn],
            queue=False
        )

        previous.click(
            render_prev_session,
            [state] + [submit_btn],
            [state, chatbot] + reference_docs + [radio, submit_btn],
            queue=False
        )

        radio.change(
            fn=vote_response,
            inputs=[state, radio, submit_btn],
            outputs=[state, pbar, submit_btn]
        )

        submit_btn.click(
            save_annotations,
            [state],
            [submit_btn, pbar],
        )
        
        demo.load(
            load_demo,
            [url_params, chatbot],
            [state, chatbot, pbar] + reference_docs,
            _js=block_js,
            queue=False
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--file_path", type=str, required=True, help="e.g., model_checkpoints/databricks_e2e_tests/vicuna13b-ft_e5-ft/test-predictions.jsonl")
    parser.add_argument("--include_idx", type=str, default="")
    parser.add_argument("--concurrency-count", type=int, default=1)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--conv_template", type=str, default="default")  # see conv_templates in gradio_dialogue.py
    args = parser.parse_args()
    logger.info(f"args: {args}")

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