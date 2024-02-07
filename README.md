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

# Generate documentations using `sphinx`:

1. make sure you installed sphinx. This includes:
      ```bash
      pip install sphinx-book-theme
      ```
2. go to the `docs` folder and run:
      ```
      make html
      ```

# Preparing data by downloading existing RQA dsets

see `scripts/data/load_hf_data.py` which should be quite self-explanatory.


# Prepare Document
```bash
python scripts/data/process_docs.py \
--document_path data/training/faire_tmp/faire_texts.jsonl \
--model_name_or_path facebook/contriever-msmarco \
--chunk_size 400 \
--chunk_overlap_size 50 \
--save_dir data/training/faire_tmp \
--save_name faire_paresed_docs
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

(for E2E test, you can enable GPT4Eval by passing the flag `--gen_gpt4eval true`.)

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
--qa_model_name_or_path model_checkpoints/databricks_flant5-xl_contriever-ft/checkpoint-700 \
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

Databricks Starling LM-7B+e5:
```bash
python scripts/test/test_e2e.py \
--qa_model_name_or_path model_checkpoints/databricks_Starling7b-1e5-train2_e5-ft/checkpoint-50 \
--assistant_prefix "GPT4 Correct Assistant" \
--user_prefix "GPT4 Correct User" \
--sep_user "<|end_of_turn|>" \
--sep_sys "<|end_of_turn|>" \
--embedding_model_name_or_path model_checkpoints/retriever_model/e5_databricks_1e4_inbatch256_chunk400_fulldoc_temp1_hard0.05_retriever_train/checkpoint-120 \
--document_path data/database/databricks/databricks_400.pkl \
--index_path data/database/databricks/databricks_400_e51e4_inbatch256_chunk400hard0.05_checkpoint120 \
--eval_data_path data/training/databricks_new/test_w_qa.jsonl \
--output_dir model_checkpoints/databricks_e2e_tests/databricks_Starling7b-1e5-train2_e5-ft
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
python scripts/test/test_e2e.py \
--qa_model_name_or_path model_checkpoints/faire_vincuna-13b-5e6-train7_e5-ft/checkpoint-350 \
# --qa_model_name_or_path model_checkpoints/faire_vincuna-7b-5e6-train7_e5-ft/checkpoint-400 \
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
Databricks contriever(DCA)
```bash
python scripts/train/retriever/train_fid_retriever.py \
--full_dataset_file_path data/training/databricks_400.pkl \
--train_file data/training/databricks_new/train_w_q_fid.json \
--eval_file data/training/databricks_new/eval_w_q_fid.json \
--model_name_or_path facebook/contriever-msmarco \
--reader_model_path google/flan-t5-xl \
--reader_temperature 0.1 \
--with_score False \
--projection True \
--n_context 50 \
--do_train True \
--do_eval True \
--learning_rate 1e-5 \
--reader_batch_size 4 \
--per_device_train_batch_size 10 \
--per_device_eval_batch_size 10 \
--metric_for_best_model eval_retr/document_recall/recall \
--max_steps 150 \
--eval_steps 5 \
--save_steps 5 \
--logging_steps 1 \
--seed 16 \
--output_dir result/model_checkpoints/fid/fid_contriever-ms_1e5_databricks_inbatch10_chunk400_fulldoc
```

Faire contriever(RPG)
```bash
python scripts/train/retriever/train_replug_retriever.py \
--full_dataset_file_path data/training/faire_400.pkl \
--train_file data/training/faire_new/train_w_qa.jsonl \
--eval_file data/training/faire_new/eval_w_qa.jsonl \
--model_name_or_path facebook/contriever-msmarco \
--lm_model_path stabilityai/stablelm-zephyr-3b \
--refresh_step 10 \
--text_maxlength 512 \
--lm_temperature 0.1 \
--retrieve_temperature 0.1 \
--num_docs 20 \
--do_train True \
--do_eval True \
--pooling_type mean \
--learning_rate 2e-5 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--metric_for_best_model eval_retr/document_recall/recall \
--max_steps 250 \
--eval_steps 5 \
--save_steps 5 \
--logging_steps 1 \
--output_dir result/model_checkpoints/replug/contriever-msmarco_mean_2e5_faire_eval_inbatch4_chunk400_fulldoc_temp0.1
```

Databricks E5(CTL)
```bash
python scripts/train/retriever/train_ctl_retriever.py \
--full_dataset_file_path data/training/databricks_400.pkl \
--train_file data/training/databricks_new/train_w_q.jsonl \
--eval_file data/training/databricks_new/eval_w_q.jsonl \
--model_path intfloat/e5-base-v2 \
--pooling_type mean \
--do_train True \
--do_eval True \
--learning_rate 1e-4 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 128 \
--hard_neg_ratio 0.05 \
--contrastive_loss inbatch_contrastive \
--metric_for_best_model eval_retr/document_recall/recall \
--max_steps 150 \
--eval_steps 5 \
--save_steps 5 \
--logging_steps 1 \
--temperature 1 \
--output_dir result/model_checkpoints/e5/e5_databricks_1e4_inbatch256_chunk400_fulldoc_temp1_hard0.05
```

Databricks BGE(CTL)
```bash
python scripts/train/retriever/train_ctl_retriever.py \
--full_dataset_file_path data/training/databricks_400.pkl \
--train_file data/training/databricks_new/train_w_q.jsonl \
--eval_file data/training/databricks_new/eval_w_q.jsonl \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--pooling_type mean \
--do_train True \
--do_eval True \
--learning_rate 1e-5 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 128 \
--hard_neg_ratio 0.05 \
--contrastive_loss inbatch_contrastive \
--metric_for_best_model eval_retr/document_recall/recall \
--max_steps 150 \
--eval_steps 5 \
--save_steps 5 \
--logging_steps 1 \
--temperature 1 \
--output_dir result/model_checkpoints/bge/bge_1e5_mean_databricks_inbatch256_chunk400_fulldoc_temp1_hard0.05
```


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
python scripts/train/qa_llm/train_w_fixed_retriever.py \
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

Databricks Starling-LM-7B-alpha + E5

```bash
torchrun --nproc_per_node=1 --master_port=20001 scripts/train/qa_llm/train_w_fixed_retriever.py \
--use_flash_attention true \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--deepspeed scripts/train/ds_config.json \
--learning_rate 1e-5 \
--num_train_epochs 2 \
--gradient_accumulation_steps 4 \
--bf16 true \
--model_name_or_path berkeley-nest/Starling-LM-7B-alpha \
--assistant_prefix "GPT4 Correct Assistant" \
--user_prefix "GPT4 Correct User" \
--sep_user "<|end_of_turn|>" \
--sep_sys "<|end_of_turn|>" \
--embedding_model model_checkpoints/retriever_model/e5_databricks_1e4_inbatch256_chunk400_fulldoc_temp1_hard0.05_retriever_train/checkpoint-120 \
--embedding_max_num_to_retrieve 3 \
--logging_steps 10 \
--eval_steps 50 \
--save_steps 50 \
--output_dir model_checkpoints/databricks_Starling7b-5e6-train7_e5-ft \
--run_group databricks_vicuna \
--train_file data/training/databricks_new/train_w_qa.jsonl \
--eval_file data/training/databricks_new/eval_w_qa.jsonl \
--test_file data/training/databricks_new/test_w_qa.jsonl \
--full_dataset_file_path data/database/databricks/databricks_400.pkl \
--full_dataset_index_path data/database/databricks/databricks_400_e51e4_inbatch256_chunk400hard0.05_checkpoint120
```

Databricks OpenChat + BGE:
```bash
python scripts/train/qa_llm/train_w_fixed_retriever.py \
--use_flash_attention true \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--deepspeed scripts/train/ds_config.json \
--learning_rate 5e-6 \
--num_train_epochs 2 \
--gradient_accumulation_steps 4 \
--bf16 true \
--model_name_or_path openchat/openchat_3.5 \
--assistant_prefix "GPT4 Correct Assistant" \
--user_prefix "GPT4 Correct User" \
--sep_user "<|end_of_turn|>" \
--sep_sys "<|end_of_turn|>" \
--embedding_model model_checkpoints/retriever_model/bge_1e5_mean_databricks_inbatch256_chunk400_fulldoc_temp1_hard0.05_train/checkpoint-60 \
--embedding_max_num_to_retrieve 3 \
--logging_steps 10 \
--eval_steps 50 \
--save_steps 50 \
--output_dir model_checkpoints/databricks_OpenChat7b-5e6-train2_bge-ft \
--run_group databricks_openchat \
--train_file data/training/databricks_new/train_w_qa.jsonl \
--eval_file data/training/databricks_new/eval_w_qa.jsonl \
--test_file data/training/databricks_new/test_w_qa.jsonl \
--full_dataset_file_path data/database/databricks/databricks_400.pkl \
--full_dataset_index_path data/database/databricks/databricks_400_bge1e5_mean_databricks_inbatch256_chunk400_ckpt60
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
python scripts/train/qa_llm/train_w_fixed_retriever.py \
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


# Serving

by default, all server logs will go under `logs` folder. Make sure this folder exists before running the commands below.
1. `python open_rqa/serve/controller.py`
2. This is where you launch your customized RQA Pipeline(s):
      ```bash
      export CUDA_VISIBLE_DEVICES=7
      python open_rqa/serve/model_worker.py \
      --document_path data/database/databricks/databricks_400.pkl \
      --index_path data/database/databricks/databricks_400_e5-base-v2 \
      --embedding_model_name_or_path intfloat/e5-base-v2 \
      --qa_model_name_or_path lmsys/vicuna-7b-v1.5 \
      --model_id simple_rqa
      ```
3. To do a quick test to see if the above is working, try `python open_rqa/serve/test_message.py --model_id simple_rqa`
4. Launch your demo page!
      ```bash
      python open_rqa/serve/gradio_web_server.py \
      --model_id simple_rqa \
      --example "What is DBFS? What can it do?" \
      --example "What is INVALID_ARRAY_INDEX?"
      ```
      where the `--model_id simple_rqa` is to let the controller know which model this demo page is for, and the `--example` are the example questions that will be shown on the demo page.

## Serving with Acceleration Framework

Here we provide an example using `vLLM`. The procedure is very similar with using `TGI`:

1. prepare and launch your `vLLM` server, hosting your generative model. For example:
      ```bash
      python -m vllm.entrypoints.api_server --model lmsys/vicuna-7b-v1.5
      ```
      this will by default host the model at `http://localhost:8000`.
2. Then all you have to do is to use `--qa_model_name_or_path vllm::http://localhost:8000/generate` instead of `--qa_model_name_or_path lmsys/vicuna-7b-v1.5` in the above section!

Currently our list of supported acceleration frameworks include `vLLM`, `SGLang`, and `TGI`.

## Serving Static Results for Evaluation

To evaluate the first 50 predictions from a prediction file (as done in our paper), run:

```bash
python open_rqa/serve/gradio_static_server.py \
--file_path model_checkpoints/databricks_e2e_tests/vicuna13b-ft_e5-ft/test-predictions.jsonl \  # prediction file output from scripts/test/test_e2e.py
--include_idx 1-50
```

we also allow a fine-grained customization (if needed) of which data to include in the evaluation. For example:

```bash
python open_rqa/serve/gradio_static_server.py \
--file_path model_checkpoints/databricks_e2e_tests/vicuna13b-ft_e5-ft/test-predictions.jsonl \  # prediction file output from scripts/test/test_e2e.py
--include_idx 1,2,5-12,14,25-63  # e.g., ask human to label data with idx [1,2] + [5,6,...,12] + [14] + [25,26,...,63]
```

and annotated data will be saved under `logs/YY-MM-DD-HH-mm-annotations.jsonl`.


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
