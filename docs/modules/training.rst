.. _train-main:

Training
========


We implement our trainers based on ``transformers`` library, which allows for configurations such as MultiGPU training.

A table here explaining with algorithm takes in what input, and what is the output.


.. note::
   Currently, our training scripts assume a single GPU setting, since our evaluation loop assumes all models are loaded on a single GPU. **To avoid unexpected behavior with multi-GPU training**, we suggest setting ``--evaluation_strategy=no`` with our current training scripts.


.. toctree::
   :maxdepth: 5
   :caption: Training
   :hidden:

   training/training_retriever.rst
   training/generator.rst