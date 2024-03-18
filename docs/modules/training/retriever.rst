.. _training-ret:

Training a Retriever
====================

Given a user query, a retriever selects *k* most relevant passages from a collection of documents.
LocalRQA implements trainers for encoders that distill from a down-stream LM and trainers that perform contrastive learning using a dataset of *<q,p>* pairs (and optionally hard negative examples):

* :ref:`training-ret-ctl`: finetune the embedding model by using contrastive learning.
* :ref:`training-ret-dca`: distill from cross-attention scores of an encoder-decoder model.
* :ref:`training-ret-rpg`: distill from an LM's probability distribution.

At a high level, we provide ready-to-use training scripts for each algorithm above. These scripts allow you to specify the training data, model, and other hyperparameters in a single command line. For instance, with CTL training:


.. code-block:: bash

    python scripts/train/retriever/train_ctl_retriever.py \
    --full_dataset_file_path <example/documents.pkl> \
    --train_file <example/train_w_q.jsonl> \
    --eval_file <example/eval_w_q.jsonl> \
    --model_name_or_path intfloat/e5-base-v2 \
    --pooling_type mean \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 128 \
    --hard_neg_ratio 0.05 \
    --metric_for_best_model eval_retr/document_recall/recall4 \
    --output_dir model_checkpoints/my_CTL_ret_model


this will finetune ``intfloat/e5-base-v2`` using the training data from ``<example/train_w_q.jsonl>``, and then save the model at ``model_checkpoints/my_CTL_ret_model``.

.. note::
    During training, our scripts will **also perform automatic retriever evaluations on the validation set**, i.e., ``<example/eval_w_q.jsonl>``. The evaluation results will be printed to the console and saved to the output directory.

    For more details on retriever evaluation, please refer to :ref:`evaluation-retriever`.


For more details on each training algorithm/script, please refer to their respective sections.


.. toctree::
   :maxdepth: 5
   :hidden:

   retriever/ctl.rst
   retriever/dca.rst
   retriever/rpg.rst