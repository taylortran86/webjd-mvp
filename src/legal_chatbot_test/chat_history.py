import json
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from pathlib import Path
from typing import List

CHAT_HISTORY_FILE = Path(__file__).parent / "chat.txt"

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
