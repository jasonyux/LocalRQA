.. LocalRQA documentation master file, created by
   sphinx-quickstart on Fri Nov 17 21:45:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LocalRQA's documentation!
===================================

LocalRQA is an open-source toolkit that enables researchers and developers to easily train, test, and deploy retrival-augmented QA  (RQA) systems using techniques from recent research. Given a collection of documents, you can use pre-built pipelines in our framework to quickly assemble an RQA system using the best off-the-shelf models. Alternatively, you can **create your own training data, train open-source models using algorithms from the latest research, and deploy your very own local RQA system**!


.. figure:: /_static/framework.png
   :align: center
   :width: 85 %
   :alt: LocalRQA Framework

   Components in LocalRQA


🌟 Why LocalRQA?
==================


LocalRQA is a toolkit designed to make researching and developing retrival-augmented QA systems more efficient and effective. It offers a variety of **training** methods curated from latest research (see :ref:`train-main`) and **automatic evaluation** metrics (see :ref:`evaluation-main`) to help users develop new RQA approaches and compare with prior work.

Moreover, LocalRQA doesn't just stop at creating these systems; it also provides tools for **deploying** these systems and improving them through real-world feedback (see :ref:`serving-main`). We offer support with popular inference acceleration frameworks such as ``vLLM`` and ``SGLang``, as well as methods to directly launch your RQA system with an interactive UI to allow users to chat with your system and provide feedback!

With a comprehensive suite of tools, LocalRQA aims to make it easier for researchers and developers to train, test, and deploy novel RQA approaches!


🚀 Getting Started!
===================

To get started with using LocalRQA, please refer to our :ref:`Installation <installation>` and :ref:`Quickstart <quickstart>` guide!


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:
   
   getting_started/installation.rst
   getting_started/quickstart.rst


.. toctree::
   :maxdepth: 1
   :caption: Modules
   :hidden:

   modules/data.rst
   modules/training.rst
   modules/evaluation.rst
   modules/serving.rst


.. toctree::
   :maxdepth: 1
   :caption: Playground
   :hidden:

   playground/custom_rqa.rst
   playground/custom_trainer.rst


.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :hidden:

   usecases/databricks.rst
   usecases/faire.rst


.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_reference/local_rqa.rst


Indices and tables
==================

* :ref:`genindex`
.. * :ref:`modindex`
* :ref:`search`
