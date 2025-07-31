from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
import json

SYSTEM_PROMPT = """
You are a system that evaluates the correctness of a response to a question.
You will be given a JSON object with the following fields: question, response, expected_answer.

The response doesn't have to exactly match all the words/context in the expected answer. It just needs to be right about
the answer to the actual question itself.

Evaluate whether the response is true or false, and return your reasoning for your decision in your final output.
Return your output in this exact JSON format:
```json
{
  "result": "true" | "false",
  "reasoning": "your response here",
}
```

EXAMPLES:
Input: ```json
{
    "question": "What is the capital of France?",
    "response": "The capital of France is Paris.",
    "expected_answer": "Paris"
}
Output: ```json
{
    "result": "true",
    "reasoning": "The response correctly identifies Paris as the capital of France."
}

Input: ```json
{
    "question": "What is 2 + 2?",
    "response": "2 + 2 equals 5.",
    "expected_answer": "4"
}
Output: ```json
{
    "result": "false",
    "reasoning": "The response incorrectly states that 2 + 2 equals 5, while the expected answer is 4."
}

"""

llm = ChatOllama(
    model="llama3.2",
    temperature=0, 
    format="json"
)

def evaluate_response(input: dict):
    """Evaluate the correctness of a response to a question."""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(input))
        ]
    )

    chain = prompt | llm
    response = chain.invoke({})
    
    return response

if __name__ == "__main__":
    # Example usage
    input = {
        "question": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "expected_answer": "Paris"
    }
    result = evaluate_response(input)
    print(result)  # Should print a JSON string with result and reasoning
    
    input = {
        "question": "What is 2 + 2?",
        "response": "2 + 2 equals 5.",
        "expected_answer": "4"
    }
    result = evaluate_response(input)
    print(result)  # Should print a JSON string with result and reasoning