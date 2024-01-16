# OpenRQA: retrieval augmented QA systems that can be easily served, trained, and evaluated

```bash
docker run -it --gpus all --shm-size=256m \
-v /local/data:/local/data \
-v /local2/data:/local2/data \
-v /proj/interaction/interaction-filer/xy2437:/proj/interaction/interaction-filer/xy2437 \  # change this
-v /local/data/xy2437/.cache:/home/docker/.cache \  # change this
-v /home/xy2437/OpenRQA:/workspace/OpenRQA \  # change this
jasonyux/nvcr_torch:23.08 bash
```

# Data Generation
Let's use GPT-3.5 to generate questions, and GPT-4-turbo to generate answers.

generate questions:
```bash
python scripts/data/doc_to_q_databricks.py \
-mode all \
-document_path data/database/databricks/databricks_400.pkl \
--prompt_model gpt-3.5-turbo \
--num_hard_negs_per_doc 2 \
--num_train_data 1200 \  # use a small number to test if it works first
--num_eval_test_data 150 \  # use a small number to test if it works first
--save_dir data/training/databricks_tmp
```
```bash
python scripts/data/doc_to_q.py \
-mode all \
-document_path data/database/faire/faire_400.pkl \
--prompt_model gpt-3.5-turbo \
--num_hard_negs_per_doc 2 \
--num_train_data 600 \
--num_eval_test_data 150 \
--save_dir data/training/faire_tmp
```

generate answers:
```bash
python scripts/data/doc_q_to_a_databricks.py \
--prompt_model gpt-4-1106-preview \
--dataset_w_q data/training/databricks_new/train_w_q.jsonl \  # generated by the previous step
--save_name train_w_qa.jsonl \
--save_dir data/training/databricks_new \
--end_data_idx 4  # a small number to test if it works
```
```bash
python scripts/data/doc_q_to_a.py \
> --prompt_model gpt-4-1106-preview \
> --dataset_w_q data/training/faire_new/train_w_q.jsonl \
> --save_name train_w_qa.jsonl \
> --save_dir data/training/faire_new \
> --end_data_idx 4
```

# Testing

E2E test GPT-3.5 + Text ada:
```bash
python scripts/test/test_e2e.py \
--qa_model_name_or_path gpt-3.5-turbo \
--embedding_model_name_or_path text-embedding-ada-002 \
--document_path data/database/databricks/databricks_400.pkl \
--index_path data/database/databricks/databricks_400_textada \
--eval_data_path data/training/databricks_new/test_w_qa.jsonl \
--output_dir model_checkpoints/databricks_e2e_tests/gpt3.5-turbo_textada
```

with GPT-4-turbo + Text ada:
```bash
python scripts/test/test_e2e.py \
--qa_model_name_or_path gpt-4-1106-preview \
--embedding_model_name_or_path text-embedding-ada-002 \
--document_path data/database/databricks/databricks_400.pkl \
--index_path data/database/databricks/databricks_400_textada \
--eval_data_path data/training/databricks_new/test_w_qa.jsonl \
--output_dir model_checkpoints/databricks_e2e_tests/gpt4-turbo_textada
```

Databricks T5-FiD + Contriever:
```bash
python scripts/test/test_e2e.py \
--qa_model_name_or_path model_checkpoints/databricks_flant5-xl_contriever-ft/checkpoint-800 \
--qa_is_fid true \
--embedding_model_name_or_path model_checkpoints/retriever_model/contriever-ms_databricks_inbatch256_chunk400_fulldoc_hard0.05_train/checkpoint-65 \
--document_path data/database/databricks/databricks_400.pkl \
--index_path data/database/databricks/databricks_400_contriever-inbatch256hard0.05checkpoint-65 \
--eval_data_path data/training/databricks_new/test_w_qa.jsonl \
--output_dir model_checkpoints/databricks_e2e_tests/flant5-xl_contriever-ft
```

Databricks Vicuna+e5:
```bash
python scripts/test/test_e2e.py \
--qa_model_name_or_path model_checkpoints/databricks_vincuna-7b-5e6-train7_e5-ft/checkpoint-400 \
--embedding_model_name_or_path model_checkpoints/retriever_model/e5_databricks_1e4_inbatch256_chunk400_fulldoc_temp1_hard0.05_retriever_train/checkpoint-120 \
--document_path data/database/databricks/databricks_400.pkl \
--index_path data/database/databricks/databricks_400_e51e4_inbatch256_chunk400hard0.05_checkpoint120 \
--eval_data_path data/training/databricks_new/test_w_qa.jsonl \
--output_dir model_checkpoints/databricks_e2e_tests/vicuna7b-ft_e5-ft
```

Faire T5-FiD + Contriever:
```bash
python scripts/test/test_e2e.py \
--qa_model_name_or_path model_checkpoints/faire_flant5-xl-2e-5_contriever-ft/checkpoint-500 \
--qa_is_fid true \
--embedding_model_name_or_path model_checkpoints/retriever_model/contriever-ms_1e4_faire_inbatch256_temp0.05_hard0.1_wd5e2_train/checkpoint-42 \
--document_path data/database/faire/faire_400.pkl \
--index_path data/database/faire/faire_400_contriever1e4_inbatch256_temp0.05_hard0.1_wd5e2_ckpt42 \
--eval_data_path data/training/faire_new/test_w_qa.jsonl \
--output_dir model_checkpoints/faire_e2e_tests/flant5-xl_contriever-ft
```

Faire Vicuna+e5:
```bash
python scripts/test/test_e2e.py --qa_model_name_or_path model_checkpoints/faire_vincuna-7b-5e6-train7_e5-ft/checkpoint-400 \
--embedding_model_name_or_path model_checkpoints/retriever_model/e5_faire_1e5_inbatch256_chunk400_fulldoc_temp1.2_hard0.05_retriever_train/checkpoint-54 \
--document_path data/database/faire/faire_400.pkl \
--index_path data/database/faire/faire_400_e51e5_inbatch256_temp1.2_hard0.05_ckpt54 \
--eval_data_path data/training/faire_new/test_w_qa.jsonl \
--output_dir model_checkpoints/faire_e2e_tests/vicuna7b-ft_e5-ft
```


```
databricks best e5 = model_checkpoints/retriever_model/e5_databricks_1e4_inbatch256_chunk400_fulldoc_temp1_hard0.05_retriever_train/checkpoint-120
databricks best contriever = model_checkpoints/retriever_model/contriever-ms_databricks_inbatch256_chunk400_fulldoc_hard0.05_train/checkpoint-65
faire best e5 = model_checkpoints/retriever_model/e5_faire_1e5_inbatch256_chunk400_fulldoc_temp1.2_hard0.05_retriever_train/checkpoint-54
faire best contriever = model_checkpoints/retriever_model/contriever-ms_1e4_faire_inbatch256_temp0.05_hard0.1_wd5e2_train/checkpoint-42
```

# Training

Databricks Fusion-in-decoder + Contriever
```bash
python scripts/train/qa_llm/train_w_gt_fid.py \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--deepspeed scripts/train/ds_config.json \
--bf16 true \
--learning_rate 1e-5 \
--num_train_epochs 7 \
--gradient_accumulation_steps 2 \
--model_name_or_path google/flan-t5-xl \
--embedding_model model_checkpoints/retriever_model/contriever-ms_databricks_inbatch256_chunk400_fulldoc_hard0.05_train/checkpoint-65 \
--embedding_max_num_to_retrieve 3 \
--logging_steps 10 \
--eval_steps 100 \
--save_steps 100 \
--output_dir model_checkpoints/databricks_flant5-xl_contriever-ft \
--run_group databricks_fid \
--train_file data/training/databricks_new/train_w_qa.jsonl \
--eval_file data/training/databricks_new/eval_w_qa.jsonl \
--test_file data/training/databricks_new/test_w_qa.jsonl \
--full_dataset_file_path data/database/databricks/databricks_400.pkl \
--full_dataset_index_path data/database/databricks/databricks_400_contriever-inbatch256hard0.05checkpoint-65
```

Databricks Vicuna + e5:
```bash
python scripts/train/qa_llm/train_w_gt.py \
--use_flash_attention true \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--deepspeed scripts/train/ds_config.json \
--learning_rate 5e-6 \
--num_train_epochs 7 \
--gradient_accumulation_steps 2 \
--bf16 true \
--model_name_or_path lmsys/vicuna-7b-v1.5 \
--embedding_model model_checkpoints/retriever_model/e5_databricks_1e4_inbatch256_chunk400_fulldoc_temp1_hard0.05_retriever_train/checkpoint-120 \
--embedding_max_num_to_retrieve 3 \
--logging_steps 10 \
--eval_steps 100 \
--save_steps 100 \
--output_dir model_checkpoints/databricks_vincuna-7b-5e6-train7_e5-ft \
--run_group databricks_vicuna \
--train_file data/training/databricks_new/train_w_qa.jsonl \
--eval_file data/training/databricks_new/eval_w_qa.jsonl \
--test_file data/training/databricks_new/test_w_qa.jsonl \
--full_dataset_file_path data/database/databricks/databricks_400.pkl \
--full_dataset_index_path data/database/databricks/databricks_400_e51e4_inbatch256_chunk400hard0.05_checkpoint120
```


Faire FID + contriever:
```bash
python scripts/train/qa_llm/train_w_gt_fid.py \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--deepspeed scripts/train/ds_config.json \
--bf16 true \
--learning_rate 2e-5 \  # since we have half the data
--gradient_accumulation_steps 2 \
--model_name_or_path google/flan-t5-xl \
--embedding_model model_checkpoints/retriever_model/contriever-ms_1e4_faire_inbatch256_temp0.05_hard0.1_wd5e2_train/checkpoint-42 \
--num_train_epochs 7 \
--embedding_max_num_to_retrieve 3 \
--logging_steps 10 \
--eval_steps 50 \
--save_steps 50 \
--output_dir model_checkpoints/faire_flant5-xl-2e-5_contriever-ft \
--run_group faire_fid \
--train_file data/training/faire_new/train_w_qa.jsonl \
--eval_file data/training/faire_new/eval_w_qa.jsonl \
--test_file data/training/faire_new/test_w_qa.jsonl \
--full_dataset_file_path data/database/faire/faire_400.pkl \
--full_dataset_index_path data/database/faire/faire_400_contriever1e4_inbatch256_temp0.05_hard0.1_wd5e2_ckpt42
```
Faire Vicuna + e5:
```bash
python scripts/train/qa_llm/train_w_gt.py \
--use_flash_attention true \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--deepspeed scripts/train/ds_config.json \
--learning_rate 5e-6 \
--num_train_epochs 7 \
--gradient_accumulation_steps 2 \
--bf16 true \
--model_name_or_path lmsys/vicuna-7b-v1.5 \
--embedding_model model_checkpoints/retriever_model/e5_faire_1e5_inbatch256_chunk400_fulldoc_temp1.2_hard0.05_retriever_train/checkpoint-54 \
--embedding_max_num_to_retrieve 3 \
--logging_steps 10 \
--eval_steps 50 \
--save_steps 50 \
--output_dir model_checkpoints/faire_vincuna-7b-5e6-train7_e5-ft \
--run_group faire_vicuna \
--train_file data/training/faire_new/train_w_qa.jsonl \
--eval_file data/training/faire_new/eval_w_qa.jsonl \
--test_file data/training/faire_new/test_w_qa.jsonl \
--full_dataset_file_path data/database/faire/faire_400.pkl \
--full_dataset_index_path data/database/faire/faire_400_e51e5_inbatch256_temp1.2_hard0.05_ckpt54
```


# References

G. Izacard, E. Grave [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282)

```
@misc{izacard2020leveraging,
      title={Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering},
      author={Gautier Izacard and Edouard Grave},
      url = {https://arxiv.org/abs/2007.0128},
      year={2020},
      publisher = {arXiv},
}
```

## Serving

by default, all server logs will go under `logs` folder. Make sure this folder exists before running the commands below.
1. `python open_rqa/serve/controller.py`
2. `export CUDA_VISIBLE_DEVICES=7 && python open_rqa/serve/model_worker.py --model-path lmsys/vicuna-7b-v1.5`
3. `python open_rqa/serve/gradio_web_server.py`