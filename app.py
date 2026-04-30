import asyncio
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil
from model import classify_image, scan_image, process_pending_images
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import timezone

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(
        lambda: asyncio.to_thread(process_pending_images),
        trigger='cron',
        hour=0,
        minute=0
    )
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
scheduler = AsyncIOScheduler(timezone=timezone.utc)

@app.post('/classify')
async def classify(file: UploadFile = File(...), threshold: int = 0):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename.split('.')[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    results = classify_image(path, threshold)
    os.remove(path)
    return results


@app.post('/scan')
async def scan(file: UploadFile = File(...), label: int = 0, user_id: int = 0):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename.split('.')[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    scan_image(path, label, user_id)
    os.remove(path)
    return {'status': 'ok'}

@app.get('/')
async def home():
    return {'status': 'ok'}