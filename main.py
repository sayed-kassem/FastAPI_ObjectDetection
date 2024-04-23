import contextlib
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import io

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from algorithm.object_detection import ObjectDetection
from models.object import Objects


 
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    object_detection.load_model()
    yield

object_detection = ObjectDetection()
    
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True,  
    allow_methods=["GET", "POST", "PUT", "DELETE"],  
    allow_headers=["*"])

async def receive(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        byte = await websocket.receive_bytes()
        try:
            queue.put_nowait(byte)
        except asyncio.QueueFull:
            pass

async def detect(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes = await queue.get()
        image = Image.open(io.BytesIO(bytes))
        objects = object_detection.predict(image)
        await websocket.send_json(objects.dict())



@app.websocket("/object-detection")
async def ws_object_detection(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    receive_task = asyncio.create_task(receive(websocket,queue=queue))
    detect_task = asyncio.create_task(detect(websocket,queue))
    try:
        done, pending = await asyncio.wait(
            {receive_task, detect_task},
            return_when= asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass


@app.post("/object-detection", response_model=Objects)
async def post_object_detection(image: UploadFile = File(...)) -> Objects:
    image_object = Image.open(image.file)
    return object_detection.predict(image_object)

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


static_files_app = StaticFiles(directory=Path(__file__).parent / "assets")
app.mount("/assets", static_files_app)