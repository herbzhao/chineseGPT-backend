from flask import Flask, request, jsonify, stream_with_context, Response
from flask_cors import CORS
import time
import json
from flask_socketio import SocketIO, emit
import simple_websocket
from backend_functions import chat
from threading import Thread

app = Flask(__name__)
CORS(app)

socketio = SocketIO(
    app, cors_allowed_origins="http://localhost:5173"
)  # add your Vue3 origin here


def background_thread():
    while True:
        data = range(1000)  # your list of data
        for d in data:
            socketio.emit("data", d)
            time.sleep(2)


@socketio.on("connect")
def test_connect():
    print("Client connected")
    emit("message", "Connected")
    # start the background thread when a client connects
    if not thread.is_alive():
        thread.start()


@app.route("/api/chatbot/stream2", methods=["GET"])
def stream_response() -> None:
    prompt = request.args.get("prompt")
    history = request.args.get("history")
    print(f"Received prompt: {request}")

    # a generator keep returning a higher count
    def generate():
        count = 0
        yield "event: myEvent\n"

        while True:
            count += 1
            time.sleep(0.3)
            data = {"message": f"{count}"}
            yield f"data: {json.dumps(data)}\n\n".encode()

    return Response(
        generate(),
        mimetype="text/event-stream",
    )


@app.route("/api/chatbot", methods=["POST"])
def send_response() -> None:
    prompt = request.json["prompt"]
    history = request.json["history"]
    print(f"Received prompt: {prompt}")
    print(f"Received history: {history}")
    # convert history to list of dict for chat function
    history = [
        {
            "role": message["author"].replace("bot", "assistant"),
            "content": message["text"],
        }
        for message in history
    ]
    # time.sleep(2)
    # return jsonify({"content": f"{prompt}!", "author": "bot", "loading": False})
    response_message = chat(
        prompt=prompt,
        history=history,
        actor="personal assistant",
        max_tokens=500,
        accuracy="medium",
        stream=True,
        session_id="test_api",
    )
    print(f"ChatGPT API reply: {response_message['content']}")
    return jsonify(
        {"content": response_message["content"], "author": "bot", "loading": False}
    )


if __name__ == "__main__":
    thread = Thread(target=background_thread)
    thread.daemon = True
    thread.start()

    socketio.run(app, debug=True, host="localhost", port=5000)
