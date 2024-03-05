Generator
=========

Given *k* retrieved passages and a user query (and optionally a chat history), generative models produce an answer conditioned on all the inputs.
LocalRQA implements supervised fine-tuning trainers that concatenate input queries with ground-truth or retrieved passages, and fusion-in-decoder trainers that process retrieved passages in parallel.
This includes trainers that: (1) supervised finetune a decoder using ground-truth *<q,a,p>* pairs; (2) supervised finetune a decoder with a frozen retriever; and (3) train an encoder-decoder with fusion-in-decoder training.
The automatic evaluation script that measures: (1)Recall@k; (2)BLEU, ROUGE and GPT-4 Eval; and (3)runtime.

Supervised finetune with ground-truth (SFT)
-------------------------------------------



Supervised finetune with a frozen retriever (SwR)
-------------------------------------------------
The sample command to finetune a generator with ``berkeley-nest/Starling-LM-7B-alpha`` as the base model by using a frozen retirever model.
::

    python scripts/train/qa_llm/train_w_fixed_retriever.py \
    --use_flash_attention true \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --deepspeed scripts/train/ds_config.json \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 4 \
    --bf16 true \
    --model_name_or_path berkeley-nest/Starling-LM-7B-alpha \
    --assistant_prefix "GPT4 Correct Assistant" \
    --user_prefix "GPT4 Correct User" \
    --sep_user "<|end_of_turn|>" \
    --sep_sys "<|end_of_turn|>" \
    --embedding_model checkpoint/to/retriever/model \
    --embedding_max_num_to_retrieve 3 \
    --logging_steps 10 \
    --eval_steps 50 \
    --save_steps 50 \
    --output_dir path/to/save/dir \
    --run_group wandb_run_group \
    --train_file path/to/train_w_qa.jsonl \
    --eval_file path/to/eval_w_qa.jsonl \
    --test_file path/to/test_w_qa.jsonl \
    --full_dataset_file_path path/to/document.pkl \
    --full_dataset_index_path path/to/index/dir/of/document



Fusion-in-decoder (FiD)
-----------------------
The sample command to finetune a generator with ``google/flan-t5-xl`` as the base model in fusion-in-decoder manner.
::

    python scripts/train/qa_llm/train_w_gt_fid.py \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --deepspeed scripts/train/ds_config.json \
    --bf16 true \
    --learning_rate 1e-5 \
    --num_train_epochs 7 \
    --gradient_accumulation_steps 2 \
    --model_name_or_path google/flan-t5-xl \
    --embedding_model checkpoint/to/retriever/model \
    --embedding_max_num_to_retrieve 3 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --output_dir path/to/save/dir \
    --run_group wandb_run_group \
    --train_file path/to/train_w_qa.jsonl \
    --eval_file path/to/eval_w_qa.jsonl \
    --test_file path/to/test_w_qa.jsonl \
    --full_dataset_file_path path/to/document.pkl \
    --full_dataset_index_path path/to/index/dir/of/document