from typing import Any, List
from llama_index import download_loader
from local_rqa.schema.document import Document
from local_rqa.text_loaders.base import BaseTextLoader
import pickle
import os


class LlamaIndexTextLoader(BaseTextLoader):
    def __init__(self, save_folder="./data", save_filename="parsed_docs", loader_func="SimpleDirectoryReader"):
        """Customized text loader by using different document_loaders provided in LangChain

        Args:
            save_folder (str, optional): The folder to save the downloaded loader function from llamaIndex and the documents file. Defaults to "data".
            save_filename (str, optional): _description_. Defaults to "parsed_docs".
            loader_func (_type_, optional): _description_. Defaults to SimpleDirectoryReader.
        """
        self.loader_func = download_loader(loader_func, custom_path=save_folder)
        self.save_folder = save_folder
        self.save_filename = save_filename
        return

    def load_data(self, *args: Any, **kwargs: Any) -> List[Document]:
        """load the documents by using the specified loader_func parameters

        Raises:
            ValueError: loader_params must be specified for the corresponding loader_function

        Returns:
            List[Document]: splitted documents
        """
        loader_parameters = kwargs.get('loader_params')
        if not loader_parameters:
            raise ValueError(f"Please specify loader_params for the corresponding loader_function {self.loader_func}")

        docs = self.loader_func(**loader_parameters).load_data()
        langchain_documents = [d.to_langchain_format() for d in docs]
        texts = self._convert_doc(langchain_documents)

        self.save_texts(texts)

        return texts
    
    def save_texts(self, texts: List[Document]):
        """save the splitted docs into pickle file

        Args:
            texts (List[Document]): docs
        """
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        save_path = os.path.join(self.save_folder, f"{self.save_filename}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(texts, f)
        print(f"Saved {len(texts)} texts to {save_path}")
        return
    

if __name__ == "__main__":
    loader_func = "SimpleDirectoryReader"
    loader_parameters = {'input_dir': "./data", "recursive": True}
    kwargs = {"loader_params": loader_parameters}
    docs = LlamaIndexTextLoader(save_folder="./data", save_filename="parsed_docs", loader_func=loader_func).load_data(**kwargs)
    print(docs)
