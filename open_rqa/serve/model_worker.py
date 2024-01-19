"""
A model worker that executes the model.
"""
import argparse
import json
import os
import uuid
import uvicorn
import torch
from typing import Optional
from transformers import set_seed, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
from open_rqa.constants import ErrorCode, SERVER_ERROR_MSG
from open_rqa.serve.base_model_worker import BaseModelWorker, app
from open_rqa.serve.gradio_rqa import GradioSimpleRQA
from open_rqa.utils import init_logger


worker_id = str(uuid.uuid4())[:8]
logger = init_logger(filename=f"logs/model_worker_{worker_id}.log")


def load_model(
    database_path: Optional[str] = None,
    document_path: Optional[str] = None,
    index_path: str = "./index",
    embedding_model_name_or_path = 'text-embedding-ada-002',
    qa_model_name_or_path: str = "lmsys/vicuna-7b-v1.5",
    qa_is_fid: bool = False,
    user_prefix: str = "USER",
    assistant_prefix: str = "ASSISTANT",
    ## model_init stuff for faster inference
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    verbose=False,
    **kwargs
):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    kwargs['low_cpu_mem_usage'] = True
    rqa = GradioSimpleRQA.from_scratch(
        database_path=database_path,
        document_path=document_path,
        index_path=index_path,
        embedding_model_name_or_path=embedding_model_name_or_path,
        qa_model_name_or_path=qa_model_name_or_path,
        qa_model_init_kwargs=kwargs,
        qa_is_fid=qa_is_fid,
        user_prefix=user_prefix,
        assistant_prefix=assistant_prefix,
        verbose=verbose,
    )
    model = rqa.get_model()
    tokenizer = rqa.get_tokenizer()
    
    try:
        context_len = model.config.max_position_embeddings
    except Exception as _:
        context_len = 4096  # default to this if cannot find it in config
    return rqa, model, tokenizer, context_len


def add_model_args(parser):
    parser.add_argument(
        "--model_id",
        type=str,
        default='simple_rqa',
        help="Name for this RQA pipeline. Used by controller.py if you are hosting multiple RQA pipelines.",
    )
    ### SimpleRQA from_scratch args
    parser.add_argument(
        "--database_path",
        type=str,
        default=None,
        help="Path to the database folder.",
    )
    parser.add_argument(
        "--document_path",
        type=str,
        default=None,
        help="Path to the chunked document pickle file.",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default='./index',
        help="Path to the index folder.",
    )
    parser.add_argument(
        "--embedding_model_name_or_path",
        type=str,
        default='intfloat/e5-base-v2',
        help="Path to the embedding model.",
    )
    parser.add_argument(
        "--qa_model_name_or_path",
        type=str,
        default='lmsys/vicuna-7b-v1.5',
        help="Path to the QA model.",
    )
    parser.add_argument(
        "--qa_is_fid",
        action="store_true",
        help="Whether the QA model is a FID model.",
    )
    parser.add_argument(
        "--user_prefix",
        type=str,
        default="USER",
        help="Prefix to add to the user input.",
    )
    parser.add_argument(
        "--assistant_prefix",
        type=str,
        default="ASSISTANT",
        help="Prefix to add to the assistant output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print out the logs.",
    )

    ### model_init stuff for faster inference
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on. Either 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--load_8bit",
        action="store_true",
        help="Load the model in 8bit mode.",
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="Load the model in 4bit mode.",
    )
    return parser


class ModelWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_name: str,
        limit_worker_concurrency: int,
        no_register: bool,
        ## SimpleRQA from_scratch args
        database_path: Optional[str] = None,
        document_path: Optional[str] = None,
        index_path: str = "./index",
        embedding_model_name_or_path = 'text-embedding-ada-002',
        qa_model_name_or_path: str = "lmsys/vicuna-7b-v1.5",
        qa_is_fid: bool = False,
        user_prefix: str = "USER",
        assistant_prefix: str = "ASSISTANT",
        ## load_model_args
        device: str = 'cuda',
        load_8bit: bool = False,
        load_4bit: bool = False,
        stream_interval: int = 2,
        conv_template: Optional[str] = None,
        embed_in_truncate: bool = False,
        seed: Optional[int] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_name,
            [],
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.rqa, self.model, self.tokenizer, self.context_len = load_model(
            database_path=database_path,
            document_path=document_path,
            index_path=index_path,
            embedding_model_name_or_path=embedding_model_name_or_path,
            qa_model_name_or_path=qa_model_name_or_path,
            qa_is_fid=qa_is_fid,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
            device=device,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            verbose=debug,
        )
        self.qa_is_fid = qa_is_fid  # this changes how input is prepared for QA
        self.device = device
        # self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            self.init_heart_beat()
        return

    def retrieve(self, params):
        if self.qa_is_fid:
            ## FiD cannot do rephrase very well as it was NOT trained to do so
            rephrased_question = params["model_input"]["question"]
        else:
            rephrased_question = self.rqa.rephrase_question_for_retrieval(**params["model_input"])
            logger.info(f'rephrased_question from {params["model_input"]["question"]} to {rephrased_question}')

        retrieved_docs = self.rqa.retrieve(
            question=rephrased_question,
        ).batch_source_documents[0]

        retrieved_docs_dict = [doc.to_dict() for doc in retrieved_docs]
        return {
            "documents": retrieved_docs_dict,
            "rephrased_question": rephrased_question,
        }


    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.tokenizer, self.model

        if self.qa_is_fid:
            if self.rqa.rqa.qa_llm.is_api_model:
                raise NotImplementedError("FID model cannot be used as an API model.")
            ## a bit more complicated for FID model
            q_w_retrieved_docs = [self.rqa.prepare_prompt_for_generation(**params["model_input"])]

            max_new_tokens = min(int(params.get("max_new_tokens", 256)), 512)
            do_sample = False  # FID model does not support sampling

            input_ids, attention_mask = self.rqa.rqa.qa_llm.encode_fid_inputs(
                q_w_retrieved_docs,
                max_length=512,  # required by FID model
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

            thread = Thread(target=model.generate, kwargs=dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
            ))
            thread.start()
        else:
            ## things are easier for decoder only model
            prompt = self.rqa.prepare_prompt_for_generation(**params["model_input"])

            temperature = float(params.get("temperature", 1.0))
            top_p = float(params.get("top_p", 1.0))
            max_context_length = self.context_len
            max_new_tokens = min(int(params.get("max_new_tokens", 256)), 512)
            do_sample = True if temperature > 0.001 else False

            if self.rqa.rqa.qa_llm.is_api_model:
                streamer = self.rqa.generate_stream_from_api(
                    input_text=prompt,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
            else:
                input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

                max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1])

                if max_new_tokens < 1:
                    yield json.dumps({"text": prompt[0] + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
                    return

                thread = Thread(target=model.generate, kwargs=dict(
                    inputs=input_ids,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                ))
                thread.start()

        stop_str = params.get("stop", "</s>")
        generated_text = ''
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        if self.device == "npu":
            import torch_npu

            torch_npu.npu.set_device("npu:0")
        self.call_ct += 1

        try:
            if self.seed is not None:
                set_seed(self.seed)
            for x in self.generate_stream(params):
                yield x
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())



def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    # parser.add_argument(
    #     "--model-names",
    #     type=lambda s: s.split(","),
    #     help="Optional display comma separated names",
    # )
    parser.add_argument("--embed-in-truncate", action="store_true")
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Print debugging messages"
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker_address = f"http://{args.host}:{args.port}"
    worker = ModelWorker(
        controller_addr=args.controller_address,
        worker_addr=worker_address,
        worker_id=worker_id,
        model_name=args.model_id,
        limit_worker_concurrency=args.limit_worker_concurrency,
        no_register=args.no_register,
        ### SimpleRQA from_scratch args
        database_path=args.database_path,
        document_path=args.document_path,
        index_path=args.index_path,
        embedding_model_name_or_path=args.embedding_model_name_or_path,
        qa_model_name_or_path=args.qa_model_name_or_path,
        qa_is_fid=args.qa_is_fid,
        user_prefix=args.user_prefix,
        assistant_prefix=args.assistant_prefix,
        ### load_model args
        device=args.device,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        stream_interval=args.stream_interval,
        conv_template=None,
        embed_in_truncate=args.embed_in_truncate,
        seed=args.seed,
        debug=args.debug,
    )
    return args, worker


if __name__ == "__main__":
    args, worker = create_model_worker()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
