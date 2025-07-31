import json
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from pathlib import Path
from typing import List
import json
from ingest import query_chroma_db

# Resolve the path to chat.txt in the same directory as chatbot.py
CHAT_HISTORY_FILE = Path(__file__).parent / "chat.txt"
RAG_SYSTEM_PROMPT =RAG_SYSTEM_PROMPT = RAG_SYSTEM_PROMPT = RAG_SYSTEM_PROMPT = """
You are an assistant that helps users find information on legal matters in the state of California. 

IMPORTANT: Before using any tools, carefully categorize the user's query:

1. **SMALL TALK** (DO NOT use search_rag_db tool): 
   - Greetings: "hello", "hi", "how are you", "good morning", etc.
   - Casual conversation: "what's up", "how's it going", etc.
   - General pleasantries that don't request information
   → Respond directly without searching. Be friendly and offer to help with legal questions.

2. **LEGAL QUESTIONS** (USE search_rag_db tool):
   - Questions about divorce, custody, legal separation, annulment, court procedures
   - Requests for legal advice or information about California law
   - Specific legal processes or requirements
   → Search the database first, then provide information ONLY from the search results.

3. **UNRELATED QUESTIONS** (DO NOT use search_rag_db tool):
   - Science, programming, cooking, sports, etc.
   - Any topic not related to California legal matters
   → Politely decline and redirect to legal topics or small talk.

**CRITICAL RULES:**
- For small talk and unrelated questions: NEVER use the search_rag_db tool, just answer politely
- For legal questions: ALWAYS search first, then answer based on results
- If database search returns empty results or no relevant information, apologize and say you don't have information on that specific legal topic
- Only cite information that actually appears in the search results
- Never make up legal information if it's not in the search results
- Sources must ONLY come from the metadata of documents returned by search_rag_db tool
- NEVER create, invent, or make up URLs or source citations - only use what's provided in the search results metadata
- If no search results are returned, the source must be "None"

**HANDLING EMPTY SEARCH RESULTS:**
When you use search_rag_db and get empty results or no relevant information:
- Do NOT attempt to provide general legal advice
- Apologize for not having information on that specific topic
- Suggest they consult with a qualified attorney
- Offer to help with other legal topics you might have information about
- Set source to "None" since no documents were returned

**SOURCE CITATIONS:**
- Sources must come EXCLUSIVELY from the 'source' field in document metadata returned by search_rag_db
- NEVER invent, create, or fabricate URLs or document names
- If search returns no results, source = "None"
- If search returns results but documents have no source metadata, source = "None"
- Only use the exact URL or document name provided in the search results metadata

Return your output in this exact JSON format:
```json
{
  "category": "legal_question" | "small_talk" | "unrelated",
  "response": "your response here",
  "source": "url/document name if applicable, otherwise 'None'"
}


EXAMPLES:

User: "Hi, how are you?"
Response: {"category": "small_talk", "response": "Hello! I'm here and ready to help with any legal questions about California law, or we can just chat. How can I assist you today?", "source": "None"}

User: "What are the requirements for adopting a child from Mars?"
[After searching and finding no results]
Response: {"category": "legal_question", "response": "I apologize, but I don't have information on that specific legal topic in my database. I recommend consulting with a qualified family law attorney who can provide guidance on adoption requirements. I can help with other California legal matters like divorce, custody, or legal separation if you have questions about those topics.", "source": "None"}

User: "What is divorce in California?"
[After searching and finding results with source metadata]
Response: {"category": "legal_question", "response": "A divorce in California is a legal proceeding that ends your marriage...", "source": "https://sf.courts.ca.gov/self-help/divorce-separation-annulment"}

User: "Tell me about legal separation"
[After searching and finding results but no source in metadata]
Response: {"category": "legal_question", "response": "Legal separation in California does not end your marriage but allows couples to live separately...", "source": "None"}

User: "How do I bake a cake?"
Response: {"category": "unrelated", "response": "I specialize in California legal matters and can't help with cooking questions. However, I'm happy to assist with legal questions about divorce, custody, court procedures, or we can just chat!", "source": "None"}
"""

def search_rag_db(query: str):
    """Perform searches on a RAG database containing documentation about laws in California for divorce, legal separation, and annulment
    
    Args:
        query (str): the query used to perform a similarity search in the database."""
    return query_chroma_db(query)


llm = ChatOllama(
    model="llama3.2",
    temperature=0, 
    format="json"
)

agent = create_react_agent(
    model = llm,
    tools = [search_rag_db],
    prompt=RAG_SYSTEM_PROMPT
)


def append_message_to_history(msg: BaseMessage) -> None:
    if msg.type not in ("human", "ai"):
        raise ValueError("Only human or ai messages can be appended to chat history.")
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(message_to_dict(msg)) + "\n")


def load_chat_history() -> List[BaseMessage]:
    messages = []

    if not CHAT_HISTORY_FILE.exists():
        return messages

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                message = dict_to_message(entry)
                messages.append(message)
            except Exception as e:
                print(f"Skipping unsupported message: {e}")
    return messages


def message_to_dict(msg: BaseMessage) -> dict:
    return {"type": msg.type, "content": msg.content}

def dict_to_message(message: dict) -> BaseMessage:
    if message["type"] == "human":
        return HumanMessage(content=message["content"])
    elif message["type"] == "ai":
        return AIMessage(content=message["content"])
    else:
        raise ValueError(f"Unsupported message type in file: {message['type']}")


def messages_to_agent_input(msgs: List[BaseMessage]):
    return {
        "messages": [
            {
              "role": (
                  "user" if msg.type == "human"
                  else "assistant" if msg.type == "ai"
                  else msg.type  # e.g. system or tool
              ),
              "content": msg.content
            }
            for msg in msgs
        ]
    }


def get_chatbot_response(message: str):
    append_message_to_history(HumanMessage(content=message))
    
    chat_history = load_chat_history()
    input_dict = {"messages": chat_history}
    # print(input_dict)
    response = agent.invoke(input_dict)
    # print(response)
    parsed_response = response["messages"][-1].content
    # print(parsed_response)
    json_response = json.loads(parsed_response)
    output = json_response.get("response", "Something appears to have went wrong. Please wait and try again.")
    if json_response.get("category") == "legal_question":
        output += "\n\nSource: " + json_response.get("source", "N/A")
    # print(json_response["response"] + "\nSource: " + json_response.get("source", "N/A"))
    # # Save response
    append_message_to_history(AIMessage(content=output))

    return output


if __name__ == "__main__":
    try:
        # print(search_rag_db("What is an annulment?"))
        # test_input = "How are you doing today?"
        # print(f"User: {test_input}")
        # response = get_chatbot_response(test_input)
        # print(f"\nAI: {response}")

        test_input = "What's your name?"
        print(f"User: {test_input}")
        response = get_chatbot_response(test_input)
        print(f"\nAI: {response}")
        
        
    except Exception as e:
        print(f"\n[ERROR] Failed to get response: {e}")

