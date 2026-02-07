# backend/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, uuid, cv2
from sam2_service import SAM2Service
from fastapi.staticfiles import StaticFiles



app = FastAPI()

app.mount("/storage", StaticFiles(directory="storage"), name="storage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "storage"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
MASK_DIR = os.path.join(BASE_DIR, "masks")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

sam2_service = SAM2Service()


def extract_frames(video_path: str, out_dir: str):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return idx


class SegmentFrameRequest(BaseModel):
    video_id: str
    frame_index: int
    pos_points: list[list[int]]
    neg_points: list[list[int]]


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    frame_dir = os.path.join(FRAME_DIR, video_id)
    os.makedirs(frame_dir, exist_ok=True)

    with open(video_path, "wb") as f:
        f.write(await file.read())

    num_frames = extract_frames(video_path, frame_dir)

    return {"video_id": video_id, "num_frames": num_frames}


@app.post("/init-video")
def init_video(video_id: str):
    frame_dir = os.path.join(FRAME_DIR, video_id)
    if not os.path.exists(frame_dir):
        raise HTTPException(404, "Frames not found")

    sam2_service.init_video(video_id, frame_dir)
    return {"status": "initialized"}


@app.post("/segment-frame")
def segment_frame(req: SegmentFrameRequest):
    try:
        mask = sam2_service.add_points(
            req.video_id,
            req.frame_index,
            req.pos_points,
            req.neg_points
        )
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    out_dir = os.path.join(MASK_DIR, req.video_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{req.frame_index:05d}.png")

    from PIL import Image
    Image.fromarray(mask).save(out_path)

    return {"status": "prompt_added", "mask": out_path}


class PropagateRequest(BaseModel):
    video_id: str

@app.post("/propagate-video-mask")
def propagate_video_mask(req: PropagateRequest):
    video_id = req.video_id

    out_dir = os.path.join(MASK_DIR, video_id)
    try:
        count = sam2_service.propagate_and_save(video_id, out_dir)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    return {"status": "done", "frames_masked": count}
