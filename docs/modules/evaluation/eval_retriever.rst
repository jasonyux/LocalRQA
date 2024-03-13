.. _evaluation-retriever:

Retriever Evaluation
====================

The retriever evaluation metrics include (1)Recall@k and nDCG@k score, and (2)runtime, which are defined in ``evaluator`` -> ``EvaluatorConfig``.
The following script ``test_retriever.py`` can be used to evaluate the retriever model after finetuning or from Huggingface. 
Notice, even though it can load and evaluate any embedding model from Huggingface, but the model would be better saved in ``SentenceTransformer``, because it will have the config file for the pooling method used to generate final embedding, otherwise it will use mean pooling by default.

::

    python ./scripts/test/test_retriever.py \
    --document_path path/to/document.pkl \
    --eval_data_path path/to/test_w_q.jsonl \
    --embedding_model_name_or_path  path/to/embedding_model \
    --index_path path/to/index \
    --output_dir path/to/output/dir \
    --test_bsz 4