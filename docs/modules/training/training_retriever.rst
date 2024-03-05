Retriever
=========

Given a user query, a retriever selects *k* most relevant passages from a collection of documents.
LocalRQA implements trainers for encoders that distill from a down-stream LM and trainers that perform contrastive learning using a dataset of *<q,p>* pairs (and optionally hard negative examples).
This includes trainers that: (1) distill from cross-attention scores of an encoder-decoder model; (2) distill from an LM's probability distribution; and (3) train using contrastive learning.
The evaluation metrics include (1)Recall@k and nDCG@k score, and (2)runtime, which are defined in ``evaluator`` -> ``EvaluatorConfig``.

Distill from Cross-Attention scores of an encoder-decoder model (DCA)
---------------------------------------------------------------------
The sample command to train a retriever with DCA method with ``google/flan-t5-xl`` as the reader model to generate Cross-Attention scores.
::
    
    python scripts/train/retriever/train_fid_retriever.py \
    --full_dataset_file_path path/to/document.pkl \
    --train_file path/to/train_w_q_fid.json \
    --eval_file path/to/eval_w_q_fid.json \
    --model_name_or_path facebook/contriever-msmarco \
    --reader_model_path google/flan-t5-xl \
    --reader_temperature 0.1 \
    --with_score False \
    --projection True \
    --n_context 50 \
    --learning_rate 1e-5 \
    --reader_batch_size 4 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --metric_for_best_model eval_retr/document_recall/recall \
    --output_dir path/to/save/dir

where train_w_q_fid.json is in the following format, where "ctxs" indicates the initial list of passages for each question retrieved::
    
    [
        {
            "id": 0,
            "question": "",
            "gold_docs":[
                {
                    "page_content": "",
                    "fmt_content": "",
                    "metadata":{}
                }
            ],
            "ctxs":[
                {
                    "text": "",
                    "title": "",
                    "score": float (optional)
                }
            ]
        }
    ]


Distill from an LM's probability distribution (RPG)
---------------------------------------------------
The sample command to train a retriever with RPG method with ``stabilityai/stablelm-zephyr-3b`` as the LM model to calculate the probability distribution.
::

    python scripts/train/retriever/train_replug_retriever.py \
    --full_dataset_file_path path/to/document.pkl \
    --train_file path/to/train_w_qa.jsonl \
    --eval_file path/to/eval_w_qa.jsonl \
    --model_name_or_path facebook/contriever-msmarco \
    --lm_model_path stabilityai/stablelm-zephyr-3b \
    --refresh_step 10 \
    --text_maxlength 512 \
    --lm_temperature 0.1 \
    --retrieve_temperature 0.1 \
    --num_docs 20 \
    --pooling_type mean \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --metric_for_best_model eval_retr/document_recall/recall \
    --output_dir path/to/save/dir


Contrastive Learning (CTL)
--------------------------
The sample command to train a retriever with CTL method.
::
    
    python scripts/train/retriever/train_ctl_retriever.py \
    --full_dataset_file_path path/to/document.pkl \
    --train_file path/to/train_w_q.jsonl \
    --eval_file path/to/eval_w_q.jsonl \
    --model_name_or_path intfloat/e5-base-v2 \
    --pooling_type mean \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 128 \
    --hard_neg_ratio 0.05 \
    --contrastive_loss inbatch_contrastive \
    --metric_for_best_model eval_retr/document_recall/recall \
    --output_dir path/to/save/dir
