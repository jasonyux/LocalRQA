RQA_PROMPT = """
You are a QA bot. Use the following pieces of context to answer the users question.
Do not mention 'this context' in your response, since the following context is only visible to you.
----------------
Context:
{formatted_documents}
----------------
{formatted_chat} {assistant_prefix}:
""".replace("\t", "").strip()