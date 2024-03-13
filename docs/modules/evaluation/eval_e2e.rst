End-to-End Evaluation
=====================


example
   .. code-block:: bash
    
      python scripts/test/test_e2e.py \
      --qa_model_name_or_path lmsys/vicuna-7b-v1.5 \
      --embedding_model_name_or_path intfloat/e5-base-v2 \
      --document_path <path/to/save/dir/filename.pkl> \
      --index_path <path/to/save/index> \
      --eval_data_path <path/to/test_w_qa.jsonl> \
      --output_dir <path/to/save/result/dir>