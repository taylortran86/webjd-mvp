from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import traceback
from pathlib import Path
import json
import uvicorn
from pydantic import BaseModel

import ingest
import chatbot

API_TOKEN = "biYapSTfLMp65cQX1vQljL04pyfIpmuSCTyOtCpEWF57K4ciBpsYH60IyaYUPBBj"

load_dotenv()
app = FastAPI()
CHAT_HISTORY_FILE = Path(__file__).parent / "chat.txt"

origins = [
    "https://59d176777d77.ngrok-free.app",
    "https://thepointergroup.retool.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

@app.get("/")
def index():
    print("Received request at /")
    return "Hello, World!"

@app.post("/api/chat")
async def chat(message: ChatMessage, authorization: str = Header(None)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=403, detail="Forbidden")

    user_message = message.message
    response = chatbot.get_chatbot_response(user_message)
    return {"status": "success", "response": response}


@app.get("/api/get_chat_history")
def get_chat_history(authorization: str = Header(None)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=403, detail="Forbidden")

    if not CHAT_HISTORY_FILE.exists():
        return []

    history = []
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                history.append(json.loads(line))
        return history
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)

