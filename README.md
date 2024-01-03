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