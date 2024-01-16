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
from transformers import set_seed
from transformers import TextIteratorStreamer
from threading import Thread
from open_rqa.constants import ErrorCode, SERVER_ERROR_MSG
from open_rqa.serve.base_model_worker import BaseModelWorker, app
from open_rqa.utils import init_logger


worker_id = str(uuid.uuid4())[:8]
logger = init_logger(filename=f"logs/model_worker_{worker_id}.log")


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
def load_model(
    model_name_or_path,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    debug=False,
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

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, **kwargs)
    context_len = 2048
    return model, tokenizer, context_len


def add_model_args(parser):
    parser.add_argument(
        "--model-path",
        type=str,
    )
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
        model_path: str,
        limit_worker_concurrency: int,
        no_register: bool,
        device: str,
        load_8bit: bool = False,
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
            model_path,
            [],
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.model, self.tokenizer, self.context_len = load_model(
            model_path,
            device=device,
            load_8bit=load_8bit,
            debug=debug,
        )
        self.device = device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            self.init_heart_beat()


    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.tokenizer, self.model

        prompt = params["prompt"]
        ori_prompt = prompt

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 512)
        do_sample = True if temperature > 0.001 else False
        stop_str = params.get("stop", "</s>")

        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1])

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
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

        generated_text = ori_prompt
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
            # for output in self.generate_stream_func(params):
            #     ret = {
            #         "text": output["text"],
            #         "error_code": 0,
            #     }
            #     if "usage" in output:
            #         ret["usage"] = output["usage"]
            #     if "finish_reason" in output:
            #         ret["finish_reason"] = output["finish_reason"]
            #     if "logprobs" in output:
            #         ret["logprobs"] = output["logprobs"]
            #     yield json.dumps(ret).encode() + b"\0"
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
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    # parser.add_argument(
    #     "--conv-template", type=str, default=None, help="Conversation prompt template."
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

    # if args.gpus:
    #     if len(args.gpus.split(",")) < args.num_gpus:
    #         raise ValueError(
    #             f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
    #         )
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.limit_worker_concurrency,
        no_register=args.no_register,
        device=args.device,
        load_8bit=args.load_8bit,
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
