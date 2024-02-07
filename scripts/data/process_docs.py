import json
import argparse

from langchain.document_loaders import *
from langchain.text_splitter import *
from transformers import AutoTokenizer

from local_rqa.text_loaders.langchain_text_loader import LangChainTextLoader


def add_parser_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--document_path", type=str,
        help=("The raw documents jsonl file in specific format: {{'text': <document content>, 'title': <document title>, 'source': <document source>}}")
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="facebook/contriever-msmarco",
        help=("The tokenizer model to measure the token length for chunk")
    )
    parser.add_argument(
        "--chunk_size", type=int, default=400,
        help="The token chunk size for each documents based on the tokenizer model"
    )
    parser.add_argument(
        "--chunk_overlap_size", type=int, default=50,
        help="The token overlap size during chunk",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Path to save the parsed documents pkl file"
    )
    parser.add_argument(
        "--save_name", type=str, default="parsed_documents",
        help="Name of the save file"
    )
    return parser


def parse_arguments(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    return args


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata['source'] = record.get("source")
    metadata['title'] = record.get("title")
    return metadata

def main(args):
    loader_func, splitter_func = JSONLoader, RecursiveCharacterTextSplitter.from_huggingface_tokenizer
    loader_parameters = {
        'file_path': args.document_path,
        'jq_schema': '.',
        'content_key': 'text',
        'json_lines': True,
        'metadata_func': metadata_func
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    splitter_parameters = {'tokenizer': tokenizer, 'chunk_size': args.chunk_size, 'chunk_overlap': args.chunk_overlap_size}
    kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
    documents = LangChainTextLoader(save_folder=args.save_dir, save_filename=args.save_name, 
                                    loader_func=loader_func, splitter_func=splitter_func).load_data(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Generate (document, question) pairs given a (chunked) document database. This can be used for generating both testing (q, doc) pairs AND training (q, doc) pairs. "
        "NOTE: for this script to work properly, we assume document.metadata['source'] is NOT EMPTY (e.g., can be the url of the unchunked document, the first level title, etc.)" )
    )
    parser = add_parser_arguments(parser)
    args = parse_arguments(parser)

    main(args)
