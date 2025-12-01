"""Prompt templates for LLM interactions using LangChain PromptTemplate."""

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# System prompt for answer generation
ANSWER_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. ONLY use information from the context provided below. Do not use any outside knowledge.
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
3. Do NOT make up or invent information. If you're not sure, say so.
4. Be concise and accurate. Cite the source sections when relevant.

Context:
{context}"""

# Human prompt for answer generation
ANSWER_HUMAN_PROMPT = "{question}"

# Create ChatPromptTemplate for answer generation
answer_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(ANSWER_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(ANSWER_HUMAN_PROMPT),
])

# System prompt for evaluation
EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator that checks answers for hallucinations and factual accuracy.

Your task is to evaluate whether the given answer:
1. Is supported by the provided context
2. Contains any information not present in the context (hallucination)
3. Accurately represents the information from the context

IMPORTANT: You must respond in the following JSON format:
{{
  "verdict": "RELIABLE" or "SUSPECTED_HALLUCINATION",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of your evaluation"
}}

Where:
- "verdict": 
  - "RELIABLE": Answer is reliable and well-supported by context
  - "SUSPECTED_HALLUCINATION": Answer has high degree of response not in context and confidence is below 0.78
- "confidence": Your confidence in the evaluation (0.0 = not confident, 1.0 = very confident)
- "reasoning": Brief explanation of your evaluation

"""

# Human prompt for evaluation
EVALUATION_HUMAN_PROMPT = """Question: {question}

Context Provided:
{context}

Answer to Evaluate:
{answer}

Please evaluate this answer and provide your assessment in the JSON format specified."""

# Create ChatPromptTemplate for evaluation
evaluation_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(EVALUATION_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(EVALUATION_HUMAN_PROMPT),
])

