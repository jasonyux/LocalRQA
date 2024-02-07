from typing import Any, List
from langchain.document_loaders import *
from langchain.text_splitter import *
from transformers import AutoTokenizer
from local_rqa.schema.document import Document
from local_rqa.text_loaders.base import BaseTextLoader
import pickle
import os
import tiktoken
import json


class LangChainTextLoader(BaseTextLoader):
    def __init__(self, save_folder="data", save_filename="parsed_docs", loader_func=DirectoryLoader, splitter_func=CharacterTextSplitter):
        """Customized text loader by using different document_loaders provided in LangChain

        Args:
            save_folder (str, optional): _description_. Defaults to "data".
            save_filename (str, optional): _description_. Defaults to "parsed_docs".
            loader_func (_type_, optional): _description_. Defaults to DirectoryLoader.
            splitter_func (_type_, optional): _description_. Defaults to CharacterTextSplitter.
        """
        self.loader_func = loader_func
        self.splitter_func = splitter_func
        self.save_folder = save_folder
        self.save_filename = save_filename
        return

    def load_data(self, *args: Any, **kwargs: Any) -> List[Document]:
        """load the documents by using the specified loader_func parameters and splitter_func parameters

        Raises:
            ValueError: loader_params and splitter_params must be specified for the corresponding loader_function and splitter_function

        Returns:
            List[Document]: splitted documents
        """
        loader_parameters = kwargs.get('loader_params')
        splitter_parameters = kwargs.get('splitter_params')
        if not loader_parameters:
            raise ValueError(f"Please specify loader_params for the corresponding loader_function {self.loader_func}")
        if not splitter_parameters:
            raise ValueError(f"Please specify splitter_params for the corresponding splitter_function {self.splitter_func}")

        docs = self.loader_func(**loader_parameters).load()
        text_splitter = self.splitter_func(**splitter_parameters)
        texts = text_splitter.split_documents(docs)
        texts = self._convert_doc(texts)

        self.save_texts(texts)

        return texts
    
    def save_texts(self, texts: List[Document]):
        """save the splitted docs into pickle file

        Args:
            texts (List[Document]): splitted docs
        """
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        save_path = os.path.join(self.save_folder, f"{self.save_filename}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(texts, f)
        print(f"Saved {len(texts)} texts to {save_path}")
        return
    

if __name__ == "__main__":
    ## Example of load document by using GoogleDriveLoader, split the docs by using CharacterTextSplitter
    loader_func, split_func = GoogleDriveLoader, CharacterTextSplitter
    loader_parameters = {'folder_id': "1KARQJvcjAbsofjyFpke-z7Udk3m_NnIF", 'recursive': True}
    splitter_parameters = {'chunk_size': 500, 'chunk_overlap': 200, 'separator': "\n\n"}
    kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
    docs = LangChainTextLoader(loader_func, split_func).load_data(**kwargs)

    ## Examples of load document from website url by using SeleniumURLLoader
    loader_func, split_func = SeleniumURLLoader, CharacterTextSplitter
    loader_parameters = {'urls': ["https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory"]}
    splitter_parameters = {'chunk_size': 500, 'chunk_overlap': 200, 'separator': "\n\n"}
    kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
    docs = LangChainTextLoader(loader_func, split_func).load_data(**kwargs)

    ## Example of local document from all .txt file under directory by using DirectoryLoader
    loader_func, split_func = DirectoryLoader, CharacterTextSplitter
    loader_parameters = {'path': "data", 'glob': "**/*.txt"}
    splitter_parameters = {'chunk_size': 500, 'chunk_overlap': 200, 'separator': "\n\n"}
    kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
    docs = LangChainTextLoader(loader_func, split_func).load_data(**kwargs)
    
    ## Example of local document from json/jsonl file by using JSONLoader
    def metadata_func(record: dict, metadata: dict) -> dict:

        metadata["subtitle"] = record.get("question")
        metadata['source'] = record.get("source")
        metadata['title'] = record.get("title")

        return metadata

    json_list_docs = json.load(open("/local2/data/shared/rqa/training/faire_raw/faire_texts.json", "r"))
    jsonl_filepath = '/local2/data/shared/rqa/training/faire_raw/faire_texts.jsonl'
    with open(jsonl_filepath, 'w') as file:
        for json_obj in json_list_docs:
            json_str = json.dumps(json_obj)
            file.write(json_str + '\n')
    loader_func, splitter_func = JSONLoader, RecursiveCharacterTextSplitter.from_huggingface_tokenizer
    loader_parameters = {
        'file_path': jsonl_filepath,
        'jq_schema': '.',
        'content_key': 'full_text',
        'json_lines': True,
        'metadata_func': metadata_func
    }
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    splitter_parameters = {'tokenizer': tokenizer, 'chunk_size': 400, 'chunk_overlap': 50}
    kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
    documents = LangChainTextLoader(save_folder="/local2/data/shared/rqa/training", save_filename="faire_400", 
                                    loader_func=loader_func, splitter_func=splitter_func).load_data(**kwargs)
