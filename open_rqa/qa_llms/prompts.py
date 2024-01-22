RQA_PROMPT = """
This is a chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers using documents from the following context.
Do not mention 'this context' in the assistant's response, since the following context is only visible to the assistant.
----------------
Context:
{formatted_documents}
----------------
{formatted_chat}{assistant_prefix}:
""".strip()


RQA_PROMPT_TRAIN = """
This is a chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers using documents from the following context.
Do not mention 'this context' in the assistant's response, since the following context is only visible to the assistant.
----------------
Context:
{formatted_documents}
----------------
{formatted_chat_w_answer}
""".strip()