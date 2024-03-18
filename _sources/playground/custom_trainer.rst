.. _play-custom-train:


Customize Model Trainers
=========================

Research in RQA is ongoing, and more and more novel training algorithms are being developed. To this end, we aim to make modules used in LocalRQA as re-usable as possible, so that you can easily use them in your own research. In this example, we will show you how to use a custom trainer in LocalRQA.


How Existing Trainers Are Implemented
-------------------------------------

If you have used any scripts we provided at ``scripts/train/``, you will notice that they all following the following style:

.. code-block:: python

    def main(model_args, data_args, logger_args, training_args):
        random.seed(0)

        ### 1. initialize model
        tokenizer = AutoTokenizer.from_pretrained(...)
        # other code omitted
        model = AutoModelForCausalLM.from_pretrained(...)
        
        ### 2. initialize datasets
        train_dset, eval_dset, test_dset = init_datasets(...)

        ### 3. initialize trainer
        eval_config = EvaluatorConfig(...)
        trainer = SupervisedTrainer(
            model=model,
            train_args=training_args,
            eval_config=eval_config,
            eval_retriever=eval_retriever,
            train_dataset=train_dset,
            eval_dataset=eval_dset,
            tokenizer=tokenizer,
        )

        ### 4. train and test
        trainer.train()
        trainer.predict(test_dset)
        return


This means that to add a new training algorithm/a trainer, you will need to:

#. implement necessary **arguments class** (e.g., ``training_args``).

   In our repo, these are defined in the ``local_rqa/trainers/*/arguments.py`` file.
#. (potentially) implement a new ``init_datasets`` function.

   In our repo, these custom datasets are defined in the ``local_rqa/trainers/*/datasets.py`` file.
#. implement a new Trainer class.

   In our repo, these are defined in the ``local_rqa/trainers/*/*_trainer.py`` file.


Adding a new Trainer
--------------------

In this example, we will show you how to add a Fusion-In-Decoder trainer (see :ref:`training-gen-fid` or ``scripts/train/qa_llm/train_w_gt_fid.py``). For more details on how fusion-in-decoder works, please refer to the `original paper <https://arxiv.org/abs/2007.01282>`_.


A high-level sketch
~~~~~~~~~~~~~~~~~~~

Let's implement this in a top-down approach. As mentioned above, we need to implement up to three components:

.. code-block:: python

    def init_datasets(data_args: DataArguments, tokenizer, tmp_output_dir: str, embedding_model):
        # some other code omitted
        train_dset = SupervisedFiDRQAwRetrieverDataset(...)
        eval_dset = SupervisedFiDRQAwRetrieverDataset(...)
        test_dset = SupervisedFiDRQAwRetrieverDataset(...)
        return train_dset, eval_dset, test_dset

    def main(model_args: ModelArguments, data_args: DataArguments, logger_args: LoggerArguments, training_args: E2EQATrainingArguments):
        random.seed(0)

        logger.info('training FiD with retrieved documents from embedding model')
        ### 1. initialize model
        tokenizer = AutoTokenizer.from_pretrained(...)
        model = FiDT5.from_t5(...)
        # some other code omitted


        ### 2. initialize datasets
        ### TODO
        train_dset, eval_dset, test_dset = init_datasets(data_args, tokenizer, training_args.output_dir, embedding_model)

        ### 3. initialize trainer
        ### TODO
        eval_config = EvaluatorConfig(...)
        trainer = SupervisedFiDTrainer(...)

        ### 4. train and test
        trainer.train()
        trainer.predict(test_dset)
        return


Implementing new arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the case of Fusion-In-Decoder training, we do **not need** to implement new training arguments, but **only a few new arguments for preparing the datasets**. Therefore, we would not need to add new classes in ``local_rqa/trainers/*/arguments.py``, but directly in the training script:

.. code-block:: python

    @dataclass
    class DataArguments:
        train_file: str = field(...)
        eval_file: str = field(...)
        test_file: str = field(...)
        # some additional fields the SupervisedFiDRQAwRetrieverDataset class might need
        max_encoder_seq_length: int = field(
            default=512,
            metadata={"help": "The maximum total input sequence length = one document + question"},
        )
        max_decoder_seq_length: int = field(
            default=256,
            metadata={"help": "The maximum total input sequence length = answer"},
        )
        embedding_model: str = field(
            default="",
            metadata={"help": "What embedding model to train with (e.g., intfloat/e5-base). If empty, train with ground truth."},
        )
        embedding_max_num_to_retrieve: int = field(
            default=3,
            metadata={"help": "Max number of documents to retrieve (excluding the gold doc), if embedding_model is none empty"},
        )

    def init_datasets(data_args: DataArguments, tokenizer, tmp_output_dir: str, embedding_model):
    # some other code omitted
        train_dset = SupervisedFiDRQAwRetrieverDataset(...)
        eval_dset = SupervisedFiDRQAwRetrieverDataset(...)
        test_dset = SupervisedFiDRQAwRetrieverDataset(...)
        return train_dset, eval_dset, test_dset

    def main(..):
        ...


Implementing the datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we need to implement datasets that can be used with Fusion-In-Decoder training. On a high level, this works by parallel encoding each context + question together, and training the decoder to learn the gold answer. Visually, we need to prepare the part before passing into the encoder as **inputs**, and the final answer as **labels**:


.. figure:: /_static/training/fid.png
   :align: center
   :width: 800px
   :alt: Fusion-in-Decoder Training

   Architecture of the Fusion-in-Decoder method. (Izacard and Grave, 2020)


Therefore, the dataset class looks like:

.. code-block:: python

    # local_rqa/trainers/qa_llm/datasets.py

    class SupervisedFiDRQAwRetrieverDataset(torch.utils.data.Dataset):
        def __init__(self, ...arguments from DataArguments):
            # some code omitted
            flattened_input, flattened_output = self.prepare_data(qa_w_doc_data)
            self.data = self.encode_data(flattened_input, flattened_output)
            return

        def prepare_data(self, qa_w_doc_data: List[Dict]):
            _necessary_fields = ['question', 'chat_history', 'gold_answer', 'gold_docs']
            assert all([field in qa_w_doc_data[0].keys() for field in _necessary_fields]), \
                f"Missing necessary fields in qa_w_doc_data: {qa_w_doc_data[0].keys()}"
            
            ## init retriever
            ## we need k passages retrieved from the retriever
            all_docs = []
            for sample in qa_w_doc_data:
                all_docs.extend([Document.from_dict(doc) for doc in sample['gold_docs']])
            retriever: BaseRetriever = self.retriever_init_fn(
                embedding_model=self.embeddings,
                documents=all_docs,
            )
            all_retrieved_docs = self.pre_retrieve_all_docs(
                retriever=retriever,
                all_questions=[sample['question'] for sample in qa_w_doc_data]
            )
            
            ## format data
            formatted_input_data = []
            formatted_output_data = []
            for i, sample in enumerate(qa_w_doc_data):
                gold_docs = [Document.from_dict(doc) for doc in sample['gold_docs']]
                chat_history = sample['chat_history']
                question = sample['question']
                gold_answer = sample['gold_answer'] + " </s>"
                retrieved_docs = all_retrieved_docs[i]
                # format dialogue
                dialogue_session = DialogueSession.from_list(chat_history)
                dialogue_session.assistant_prefix = self.assistant_prefix
                dialogue_session.user_prefix = self.user_prefix
                dialogue_session.add_user_message(question)
                # since FiD is encoder decoder, input do NOT include the answer
                fmt_dialogue = dialogue_session.to_string()

                ### prompt with retrieved documents
                # fid does it in parallel
                to_include_docs = self._combine_retrieved_docs(gold_docs, retrieved_docs)
                fid_input = []
                for doc in to_include_docs:
                    # since FiD is encoder decoder, input do NOT include the answer
                    prompt = RQA_PROMPT.format(
                        formatted_documents = doc.fmt_content,
                        formatted_chat = fmt_dialogue,
                        assistant_prefix = self.assistant_prefix,
                    )
                    fid_input.append(prompt)
                formatted_input_data.append(fid_input)
                # fid output
                formatted_output_data.append(gold_answer)
            # print one example data
            logger.info("Example formatted data:")
            logger.info(formatted_input_data[0])
            logger.info(formatted_output_data[0])
            return formatted_input_data, formatted_output_data
        
        ## some other code omitted

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]


Implementing the Trainer
~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we need to implement the trainer. In case of fusion-in-decoder training, part of the logic is already included when we define the ``FiDT5`` model. But in general, it simply comes down to:

#. start from huggingface's ``Trainer`` class
#. override the ``compute_loss`` method to customize how to train the model
#. (optionally) customize the ``evaluation_loop`` method to change the evaluation logic


.. code-block:: python

    # local_rqa/trainers/qa_llm/supervised_fid_trainer.py

    class SupervisedFiDTrainer(Trainer):
        def __init__(...):
            # some code omitted
            return

        def compute_loss(self, model, inputs, return_outputs=False):
            # essentially its a simple forward pass by the FiDT5 model
            loss = model(
                input_ids=inputs['input_ids'].to(model.device),
                attention_mask=inputs['attention_mask'].to(model.device),
                labels=inputs['labels'].to(model.device),
                return_dict=False
            )[0]
            return loss
        
        # some other code omitted
        
        def evaluation_loop(
            self,
            dataloader,
            description: str,
            prediction_loss_only = None,
            ignore_keys = None,
            metric_key_prefix: str = "eval",
        ) -> EvalLoopOutput:
            ## end-to-end evaluation with a retriever
            
            # some code omitted
            wrapped_model_for_eval = self.wrap_model_for_eval(
                retriever=self.eval_retriever,
                qa_model=model,
                tokenizer=self.tokenizer,
            )

            loaded_eval_data = self._load_eval_data(self.args.eval_data_path)
            evaluator = E2EEvaluator(
                config=self.evaluator_config,
                test_data=loaded_eval_data,
            )
            performance, predictions = evaluator.evaluate(wrapped_model_for_eval, prefix=metric_key_prefix)
            output.metrics.update(performance)

            # some code omitted
            return output

        def wrap_model_for_eval(
            self,
            retriever: BaseRetriever,
            qa_model,
            tokenizer,
        ) -> SimpleRQA:
            wrapped_model = SimpleRQA.from_huggingface_fid(
                retriever=retriever,
                qa_model=qa_model,
                qa_tokenizer=tokenizer,
                user_prefix="USER",
                assistant_prefix="ASSISTANT",
            )
            return wrapped_model

In this case, ``compute_loss`` is quite simple, as FiD training comes down to some modification of the encoder-decoder architecture (to take in data from this new format), followed by a normal forward pass to compute the loss.

During ``evaluation_loop``, by default we re-use the ``E2EEvaluator`` class to perform end-to-end evaluation with a retriever. This can also be done very easily **because** ``wrap_model_for_eval`` **simply assembles an** ``SimpleRQA`` **class, and** ``SimpleRQA`` **already supports loading FiD models as a QA model**. For more details of how this works, please refer to the ``HuggingFaceFiDQAModel`` wrapper class under ``local_rqa/qa_llms/huggingface.py``.