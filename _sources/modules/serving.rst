.. _serving-main:

Serving
=======

LocalRQA provides methods to 1) deploy your RQA system with a simple model backend, and 2) launch an interactive UI for human evaluation or free chat. This can be used to showcase your RQA system to the public, or to collect human feedback to further improve your system using techniques such as RLHF.


Launchers
---------


Currently, we provide methods to launch your RQA system in two different ways.

**Static evaluation**: users directly evaluate the quality of pre-generated responses (e.g., computed from a test set). See :ref:`serving-human-eval` for more details.

.. figure:: /_static/serving/human_eval_ui.png
   :scale: 60 %
   :alt: Static Evaluation UI

   Static Evaluation


**Interactive chat**: users can chat with a system and rate the correctness/helpfulness of each response. See :ref:`serving-interactive-eval` for more details.

.. figure:: /_static/serving/interactive_ui.png
   :scale: 60 %
   :alt: Interactive Chat UI

   Interactive Chat


Acceleration Frameworks
-------------------------

We also integrate with several inference acceleration frameworks to speed up model's retrieval/answer generation. Currently, we support the following frameworks:

To speed up retrieval:

* `FAISS <https://github.com/facebookresearch/faiss>`_


To speed up text generation:

* `Text Generation Inference <https://github.com/huggingface/text-generation-inference>`_
* `vLLM <https://github.com/vllm-project/vllm>`_
* `SGLang <https://github.com/sgl-project/sglang>`_


See :ref:`serving-acc-frameworks` for more details on how to use them with the serving methods mentioned above.


.. toctree::
    :maxdepth: 1
    :caption: Serving
    :hidden:

    serving/human_eval.rst
    serving/interactive.rst
    serving/acc_frameworks.rst