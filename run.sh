CUDA_VISIBLE_DEVICES=4 python ./scripts/train/retriever/train.py \
--documents_path /local2/data/shared/rqa/training/databricks/eval_documents.pkl \
--train_file /local2/data/shared/rqa/training/databricks/train.pkl \
--eval_file /local2/data/shared/rqa/training/databricks/eval.pkl \
--per_device_train_batch_size 256 \
--search_algo inner_product \
--hard_neg_ratio 0.05 \
--contrastive_loss inbatch_contrastive \
--max_steps 400 \
--eval_steps 50 \
--save_steps 50 \
--logging_steps 10 \
--exp_name contriever-ms_databricks_inbatch256_hard0.05_test