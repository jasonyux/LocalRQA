REPHRASE_QUESTION_PROMPT = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History: {chat_history_str}
Follow Up Input: {question}{eos_token}
Standalone question:
""".replace(
    " " * 4, ""
).strip()