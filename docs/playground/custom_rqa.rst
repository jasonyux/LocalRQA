.. _play-custom-rqa:


Customize your RQA system
==========================

Our ``SimpleRQA`` simply takes three ingredients together:

- a document and index database
- a retrieval model
- a QA model

At a high level, when you call the ``qa`` method the following happens:

.. code-block:: python

    rqa = SimpleRQA.from_scratch(...)
    ## rqa.components = [retriever, qa_llm, answer_guardrail]
    ## SimpleRQA by default adds in an answer guardrail that does NOOP for now
    
    response = rqa.qa(
        batch_questions=['What is DBFS?'],
        batch_dialogue_session=[DialogueSession()],
    )
    ## 1. call retriever.run              with input {batch_questions, batch_dialogue_session}, output {batch_source_documents}
    ## 2. call qa_llm.run                 with input {batch_questions, batch_dialogue_session, batch_source_documents}, output {batch_answers}
    ## 3. call answer_guardrail.guardrail with input {batch_questions, batch_dialogue_session, batch_source_documents, batch_answers}, output {RQAOutput}


This means that to **customize your own RQA system, simply modify this component cascade!** To illustrate this, let's walk through an example of adding a custom answer guardrail module using OpenAI's moderation API.


Implementing a Custom Guardrail
-------------------------------

First, let's implement a simple ``component`` that can be added to the RQA system.

.. note::

    In principle, you do not need to stick with using the ``SimplRQA`` class. In general, other functionalities of our code base rely on **the parent** ``RQAPipeline`` **class**. We use ``SimpleRQA`` here for simplicity.


The ``Component`` class
~~~~~~~~~~~~~~~~~~~~~~~


Under the hood, each ``component`` inherits the ``Component`` class, which instructs the RQA pipeline of the following:

.. code-block:: python

    class BaseAnswerGuardrail(Component):
        ### 1. what input data are required to run the component
        run_input_keys = [
            "batch_questions",
            "batch_source_documents",
            "batch_dialogue_session",
            "batch_answers",
        ]

        @abstractmethod
        def guardrail(
            self,
            batch_questions: List[str],
            batch_source_documents: List[List[Document]],
            batch_dialogue_session: List[DialogueSession],
            batch_answers: List[str],
        ) -> RQAOutput:
            ...

        ### 2. the 'forward' pass when this component is called
        def run(self, *args, **kwargs):
            return self.guardrail(*args, **kwargs)


In this case, because you might to check if the generated answer is faithful to the input question and retrieved documents, the ``run_input_keys`` are set to include both of them. When the ``run`` method is called, the following happens:

#. say the previous component returned the following output:

   .. code-block:: python

        {
            "batch_questions": ['What is DBFS?'],
            "batch_source_documents": [ [Document(...)] ],
            "batch_dialogue_session": [DialogueSession()],
            "batch_answers": ['DBFS is ...'],
            "additional_data": "..."
        }
    
#. ``RQAPipeline`` will then choose the relevant data according to ``run_input_keys`` of this component (i.e., remove ``additional_data``), and call ``run`` with:

   .. code-block:: python
   
      self.run(
          batch_questions=['What is DBFS?'],
          batch_source_documents=[ [Document(...)] ],
          batch_dialogue_session=[DialogueSession()],
          batch_answers=['DBFS is ...'],
      )
    
#. the output will then be updated before being passed to the next component. Let's say this component changed the answer to "I'm sorry, I cannot answer that question." and removed the source documents. Then *all available data for the next component* will be:

   .. code-block:: python
   
        {
            "batch_questions": ['What is DBFS?'],
            "batch_source_documents": [],
            "batch_dialogue_session": [DialogueSession()],
            "batch_answers": ["I'm sorry, I cannot answer that question."],
            "additional_data": "..."
        }


Implementing an ``OpenAIModeration`` component
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we can implement a simple answer guardrail that checks if the answer violates OpenAI moderation API. To do this, simply implement the ``guardrail`` method and inherit the ``BaseAnswerGuardrail`` class by **defining its** ``run_input_keys`` **and** ``run`` **method**:


.. code-block:: python
    
    from typing import List
    from local_rqa.schema.document import Document
    from local_rqa.schema.dialogue import DialogueSession, RQAOutput
    from local_rqa.guardrails.base import BaseAnswerGuardrail
    import os
    import requests


    class OpenAIModeration(BaseAnswerGuardrail):
        """checks if the answer violates OpenAI moderation API."""
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

        def _moderate_single(self, text: str) -> bool:
            url = "https://api.openai.com/v1/moderations"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.OPENAI_API_KEY
            }
            text = text.replace("\n", "")
            json_data = {'input': text}
            try:
                ret = requests.post(url, headers=headers, json=json_data, timeout=5)
                flagged = ret.json()["results"][0]["flagged"]
            except requests.exceptions.RequestException as _:
                flagged = False
            except KeyError as _:
                flagged = False
            return flagged

        def guardrail(
            self,
            batch_questions: List[str],
            batch_source_documents: List[List[Document]],
            batch_dialogue_session: List[DialogueSession],
            batch_answers: List[str],
        ) -> RQAOutput:
            checked_answers = []
            checked_source_documents = []
            for idx, answer in enumerate(batch_answers):
                if self._moderate_single(answer):
                    checked_answers.append("I'm sorry, I cannot answer that question.")
                    checked_source_documents.append([])
                else:
                    checked_answers.append(answer)
                    checked_source_documents.append(batch_source_documents[idx])

            return RQAOutput(
                batch_answers=checked_answers,
                batch_source_documents=checked_source_documents,
                batch_dialogue_session=batch_dialogue_session,
            )

        def run(self, *args, **kwargs):
            return self.guardrail(*args, **kwargs)


This will take the ``batch_answers`` and check if they violate `OpenAI moderation API <https://platform.openai.com/docs/guides/moderation>`_. If the answer is flagged, it will be replaced with "I'm sorry, I cannot answer that question." and the source documents will be removed.


Adding the Guardrail to ``SimpleRQA``
-------------------------------------

Finally, we can add this guardrail to the ``SimpleRQA`` system. Since this inherits from the ``Component`` class, ``SimpleRQA`` will understand how to call it. Here's how you can add it to the system:


.. code-block:: python

    rqa = SimpleRQA.from_scratch(
        document_path="example/demo/databricks_web.pkl",
        index_path="example/demo/index",
        embedding_model_name_or_path="intfloat/e5-base-v2",
        qa_model_name_or_path="lmsys/vicuna-7b-v1.5"
    )
    guardrail = OpenAIModeration()
    rqa.components.append(guardrail)

    # run QA: retrieval -> QA -> guardrail
    response = rqa.qa(
        batch_questions=['What is DBFS?'],
        batch_dialogue_session=[DialogueSession()],
    )
    # if the answer violates OpenAI moderation API,
    # the answer will be replaced with "I'm sorry, I cannot answer that question."