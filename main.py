from fastapi import FastAPI, WebSocket
import time

app = FastAPI()


@app.get("/")
def root():
    return {"msg": "welcome"}


@app.get("/api")
def root_test_1():
    return {"msg": "test_1"}


@app.get("/api/test")
def root_test_2():
    return {"msg": "test2"}


async def example_generator(data):
    for i in range(10):
        yield f"{data} + {i}"
        time.sleep(0.1)


# https://www.starlette.io/websockets/
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()
    print(f"Received data: {data}")
    # send generator data to client
    async for value in example_generator(data["message"]):
        await websocket.send_json({"data": value})
    await websocket.send_json({"data": "completed"})
    await websocket.close(code=1000, reason=None)
