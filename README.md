# LocalRQA

:books: <a href="https://arxiv.org/abs/2403.00982">Paper</a> • :rocket: <a href="#getting-started">Getting Started</a> • :pencil2: <a href="demo/README.md">Documentations</a>

LocalRQA is an open-source toolkit that enables researchers and developers to easily train, test, and deploy retrieval-augmented QA (RQA) systems using techniques from recent research. Given a collection of documents, you can use pre-built pipelines in our framework to quickly assemble an RQA system using the best off-the-shelf models. **Alternatively, you can create your own training data, train open-source models using algorithms from the latest research, and deploy your very own local RQA system!**

## Installation

You can either install the package from GitHub or use our pre-built Docker image.

**From GitHub**

First, clone our repository

```bash
git clone https://github.com/jasonyux/LocalRQA
cd LocalRQA
```


Then run

```bash
pip install --upgrade pip
pip install -e .
```


**From Docker**

```bash
docker pull jasonyux/localrqa
docker run -it jasonyux/localrqa
```


our code base is located at ``/workspace/LocalRQA``.

## Getting Started

In essence, a retrieval-augmented QA (RQA) system is composed of two parts:

- a document database (a collection of documents)
- a embedding model + a generative model

As a quick start, we provide a simple example to **obtain a document database from a website**, and build an RQA system using **off-the-shelf models** from huggingface. As a reference, the full example code can be found in ``demo.py`` script at the root of the repository.

#### 1. Prepare Data

LocalRQA integrates with frameworks such as LangChain and LlamaIndex to easily ingest text data in various formats, such as JSON data, HTML data, data from Google Drive, etc. For example, you could load data from a website using ``SeleniumURLLoader`` from ``langchain``, then save and parse them into a collection of documents (``docs``):

```python
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import CharacterTextSplitter
from local_rqa.text_loaders.langchain_text_loader import LangChainTextLoader

# specify how to load the data and how to chunk them
loader_func, split_func = SeleniumURLLoader, CharacterTextSplitter
loader_parameters = {'urls': ["https://docs.databricks.com/en/dbfs/index.html"]}
splitter_parameters = {'chunk_size': 400, 'chunk_overlap': 50, 'separator': "\n\n"}
kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}

# load the data, chunk them, and save them
docs = LangChainTextLoader(
      save_folder="example",  # where data is saved
      save_filename="documents.pkl",
      loader_func=loader_func,
      splitter_func=split_func
).load_data(**kwargs)
```

this list of documents (``docs``) is now your **document database**, which will be used to create an embedding index for the RQA system.

#### 2. Build an RQA System

Given a path to a document database (see above), we can directly use ``SimpleRQA`` to 1) create and save an embedding index if ``example/index`` is empty, 2) plugin an embedding model and a generative model, and 3) run QA!

```python
from local_rqa.pipelines.retrieval_qa import SimpleRQA
from local_rqa.schema.dialogue import DialogueSession

rqa = SimpleRQA.from_scratch(
      document_path="<example/documents.pkl>",
      index_path="<example/index>",
      embedding_model_name_or_path="intfloat/e5-base-v2",  # embedding model
      qa_model_name_or_path="lmsys/vicuna-7b-v1.5"  # generative model
)
response = rqa.qa(
      batch_questions=['What is DBFS?'],
      batch_dialogue_session=[DialogueSession()],
)
print(response.batch_answers[0])
# DBFS stands for Databricks File System, which is a ...
```

## Train your RQA System

Different from other frameworks, LocalRQA features methods to locally train/test your RQA system using methods curated from latest research. We thus provide a large collection of training and (automatic) evaluation methods to help users easily develop new RQA systems. For a list of supported training algorithms, please refer to ..

As a simple example, below is an example script using simple SFT to train ``mistralai/Mistral-7B-Instruct-v0.2``:

```bash
python scripts/train/qa_llm/train_w_gt.py \
--use_flash_attention true \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--deepspeed scripts/train/ds_config.json \
--learning_rate 5e-6 \
--num_train_epochs 2 \
--gradient_accumulation_steps 2 \
--bf16 true \
--model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
--assistant_prefix [/INST] \
--user_prefix "<s>[INST]" \
--sep_user " " \
--sep_sys "</s>" \
--eval_embedding_model intfloat/e5-base-v2 \
--logging_steps 10 \
--eval_steps 30 \
--save_steps 30 \
--output_dir model_checkpoints/databricks_exp \
--run_group databricks \
--train_file example/databricks/processed/train_w_qa.jsonl \
--eval_file example/databricks/processed/eval_w_qa.jsonl \
--test_file example/databricks/processed/test_w_qa.jsonl \
--full_dataset_file_path example/databricks/database/databricks.pkl \
--full_dataset_index_path example/databricks/database/index
```

## Deploy your RQA System

LocalRQA provides two methods to showcase your RQA system to external users: 1) **a static evaluation webpage** where users can directly assess the system’s performance using a test dataset, or 2) **an interactive chat webpage** where users can chat with the system and provide feedback for each generated response.

#### Static Evaluation Webpage

To evaluate the first 50 predictions from a prediction file (e.g., produced by our training/evaluation script), run:

```bash
python local_rqa.serve.gradio_static_server.py \
--file_path <path/to/yourtest-predictions.jsonl> /
--include_idx 1-50
```

#### Interactive Chat Webpage

To host your model and launch an interactive chat webpage, you will need to start a model worker (hosting your models), and a model controller (dealing with user requests):

1. run `python open_rqa.serve.controller.py`
2. launch your customized RQA system(s):
      ```bash
      export CUDA_VISIBLE_DEVICES=0
      python open_rqa.serve.model_worker.py \
      --document_path example/databricks/database/databricks.pkl \
      --index_path example/databricks/database/e5-v2-index \
      --embedding_model_name_or_path intfloat/e5-base-v2 \
      --qa_model_name_or_path lmsys/vicuna-7b-v1.5 \
      --model_id simple_rqa
      ```
3. To do a quick test to see if the above is working, try `python local_rqa.serve.test_message.py --model_id simple_rqa`
4. Launch your demo page!
      ```bash
      python local_rqa.serve.gradio_web_server.py \
      --model_id simple_rqa \
      --example "What is DBFS? What can it do?" \
      --example "What is INVALID_ARRAY_INDEX?"
      ```
      where the `--model_id simple_rqa` is to let the controller know which model this demo page is for, and the `--example` are the example questions that will be shown on the demo page.


For more details on model serving, please refer to our documentation website.



