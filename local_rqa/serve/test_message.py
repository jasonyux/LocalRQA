import argparse
import json
import requests

from local_rqa.serve.gradio_dialogue import default_conversation


def main(args):
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address",
            json={"model": args.model_id}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    conv = default_conversation.clone()
    conv.add_user_message(args.message)
    prompt = conv.get_prompt()

    headers = {"User-Agent": "LocalRQA Client"}
    pload = {
        "model": args.model_id,
        "model_input": {
            "retrieved_docs": [],
            "question": prompt,
            "history": [],
        },
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "stop": conv._session.sep_sys,
    }
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=pload,
        stream=True
    )

    print(prompt.replace(conv._session.sep_sys, "\n"), end="")
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"].split(conv._session.sep_sys)[-1]
            print(output, end="\r")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model_id", type=str, default="simple_rqa")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--message", type=str, default= "Tell me a joke with more than 20 words.")
    args = parser.parse_args()

    main(args)