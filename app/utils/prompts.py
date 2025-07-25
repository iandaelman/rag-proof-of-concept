RETRIEVE_DOCUMENTS_OR_RESPOND_PROMPT = """
This method decides whether to call the retriever tool or respond directly.

If the user's question is trivial, respond directly. Just respond directly. Do not show your reasoning or thinking process.
If the question is non-trivial, use the retriever tool to generate a response.

Given the user's question:  
"{message}"

Determine whether the question is trivial. 
"""

CLASSIFY_QUESTION_PROMPT = (
    "You are a classifier determining whether a user question is trivial or non-trivial.\n"
    "A trivial question can be answered directly without needing to retrieve external documents.\n"
    "A non-trivial question requires additional context or information from documents to answer properly.\n\n"
    "Here is the user question: {message}\n\n"
    "Based on the complexity and information need of the question, respond with a single word:\n"
    "'trivial' if it can be answered directly, or 'non-trivial' if it needs document retrieval."
)


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
    "Rewrite the last question to make it clearer and more specific, without changing its original meaning or intent.\n"
    "- If the question is in Dutch, rewrite and translate it to French.\n"
    "- If the question is in French, rewrite and translate it to English.\n"
    "- Do not repeat or restate earlier questions exactly.\n"
    "- If the last question is very similar to a previous one, improve it by adding useful clarifications or context relevant to Myminfin IT support.\n"
    "- Return only the rewritten question, nothing else."
)

GENERATE_ANSWER_PROMPT = (
    "You are a helpful assistant supporting users with their MyMinfin IT-related questions.\n"
    "Based on the following context, please provide a clear and complete answer.\n"
    "If the answer is not available in the context, kindly let the user know that you don't have enough information.\n"
    "Always respond in the same language this question {question} is asked, even if the context is in a different language.\n\n"
    "Question: {question}\n"
    "Context: {context}"
)