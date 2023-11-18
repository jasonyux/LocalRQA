from open_rqa.pipelines.retrieval_qa import BaseRQA, AutoRQA
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.guardrails.base import NoopAnswerGuardrail
from open_rqa.retrievers.base import BaseRetriever


if __name__ == "__main__":
    ###### Manual usage of RQA ######
    # TODO: a quick way to load data into retriever
    documents = SimpleDirectoryReader("data").load_data()
    retriever: BaseRetriever = None

    # pick a QA model
    qa_llm = HuggingFaceQAModel()
    # pick an answer guardrail

    answer_guardrail = NoopAnswerGuardrail()

    # QA!
    rqa = BaseRQA(
        retriever=retriever,
        qa_llm=qa_llm,
        answer_guardrail=answer_guardrail
    )

    response = rqa.qa(
        batch_questions=["What is the capital of the United States?"],
        batch_dialogue_history=[DialogueSession()],
    )
    print(response)


    ##### Auto usage of RQA ######
    documents = SimpleDirectoryReader("data").load_data()
    rqa = AutoRQA(
        documents,
        retriever="e5",
        qa_llm="llama-2",
    )

    response = rqa.qa(
        batch_questions=["What is the capital of the United States?"],
        batch_dialogue_history=[DialogueSession()],
    )
    print(response)