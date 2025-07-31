from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv
import traceback
from pathlib import Path
import json

import ingest
import chatbot

API_TOKEN = "biYapSTfLMp65cQX1vQljL04pyfIpmuSCTyOtCpEWF57K4ciBpsYH60IyaYUPBBj"

load_dotenv()
app = Flask(__name__)
CHAT_HISTORY_FILE = Path(__file__).parent / "chat.txt"
CORS(app, resources={r"/api/*": {"origins": [
    "https://59d176777d77.ngrok-free.app",
    "https://thepointergroup.retool.com"
]}})


@app.route('/')
def index():
    print("Received request at /")
    return "Hello, World!"

@app.route("/api/chat", methods=["POST"])
def chat():
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {API_TOKEN}":
        abort(403)

    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400
    user_message = data.get("message", "")
    response = chatbot.get_chatbot_response(user_message)
    return jsonify({"status": "success", "response": response})


@app.route("/api/get_chat_history", methods=["GET"])
def get_chat_history():
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {API_TOKEN}":
        abort(403)

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
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
    # app.run(debug=True, port=5050)

