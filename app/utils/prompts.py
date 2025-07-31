RETRIEVE_DOCUMENTS_OR_RESPOND_PROMPT = """
This method decides whether to call the retriever tool or respond directly.

If the user's question is trivial, respond directly. Just respond directly. Do not show your reasoning or thinking process.
If the question is non-trivial, use the retriever tool to generate a response.

Given the user's question:  
"{message}"

Determine whether the question is trivial. 
"""

GRADE_DOCUMENTS_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here are the retrieved documents: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

REWRITE_QUESTION_PROMPT = (
    "You are assisting with improving a user's question related to Myminfin IT support.\n"
    "Conversation history so far (most recent last):"
    "\n ------- \n"
    "{questions}"
    "\n ------- \n"
    "Original question:"
    "\n ------- \n"
    "{original_question}"
    "\n ------- \n"
    "Rewrite the last question to make it short, clear, and easy to match with relevant documents in a vector database.\n"
    "- Keep the meaning and intent exactly the same.\n"
    "- Use simple, direct wording without adding unnecessary details.\n"
    "- If the question is in Dutch, rewrite and translate it to French.\n"
    "- If the question is in French, rewrite and translate it to English.\n"
    "- Avoid repeating previous questions word-for-word.\n"
    "- If it’s similar to a previous question, add only minimal context needed for distinction.\n"
    "- Return only the rewritten question."
)

GENERATE_ANSWER_PROMPT = (
    "You are a helpful assistant supporting users with their MyMinfin IT-related questions.\n"
    "Based on the following context, please provide a clear and complete answer.\n"
    "If the answer is not available in the context, kindly let the user know that you don't have enough information.\n"
    "If the context contains a source, always mention it in your answer as the reference.\n"
    "Always respond in the same language this question {question} is asked, even if the context is in a different language.\n\n"
    "Question: {question}\n"
    "Context: {context}"
)