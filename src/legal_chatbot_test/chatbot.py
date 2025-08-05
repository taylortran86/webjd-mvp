import json
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List
from legal_chatbot_test.chat_history import append_message_to_history, load_chat_history
from legal_chatbot_test.agent import agent





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

