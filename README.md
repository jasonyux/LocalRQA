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

Let's use GPT-3.5 to generate questions, and GPT-4-turbo to generate answers.

generate questions:
```bash
python scripts/data/doc_to_q_databricks.py \
-mode all \
-document_path data/training/databricks_sources_official_short.pkl \
--prompt_model gpt-3.5-turbo \
--num_hard_negs_per_doc 2 \
--num_train_data 10 \  # a small number to test if it works
--num_eval_test_data 10 \  # a small number to test if it works
--save_dir data/training/databricks_new
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

# Tests

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