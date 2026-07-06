# API endpoints and background task workers
import os
import uuid
import glob
import re
import shutil
import tempfile
import librosa
import soundfile as sf
import gdown
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from app.services import core_processing_pipeline, generate_clinical_report

api_router = APIRouter()

# In-memory database for background task tracking
BACKGROUND_TASKS_DB = {}

class DrivePayload(BaseModel):
    drive_url: str

def process_session_worker(task_id: str, session_files_list: list):
    """Executes ML logic and updates task status globally."""
    try:
        session_timeline, pipeline_latency = core_processing_pipeline(session_files_list)
        if not session_timeline:
            BACKGROUND_TASKS_DB[task_id] = {"success": False, "status": "failed", "detail": "Empty session track elements."}
            return
            
        html_report, llm_latency = generate_clinical_report(session_timeline)
        total_latency = pipeline_latency + llm_latency

        BACKGROUND_TASKS_DB[task_id] = {
            "success": True,
            "status": "completed",
            "session_statistics": {
                "total_chunks_processed": len(session_timeline),
                "total_processing_latency_sec": total_latency
            },
            "detailed_timeline": session_timeline,
            "clinical_insight_report": html_report
        }
    except Exception as e:
        BACKGROUND_TASKS_DB[task_id] = {"success": False, "status": "failed", "detail": str(e)}

def batch_upload_worker(task_id: str, files_payload: list):
    """Parses raw upload bytes into standard audio matrices."""
    session_files_list = []
    for f in files_payload:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(f["bytes"])
            tmp_p = tmp.name
        try:
            samples, sr = sf.read(tmp_p)
        except Exception:
            samples, sr = librosa.load(tmp_p, sr=16000, mono=True)
        finally:
            if os.path.exists(tmp_p): os.remove(tmp_p)
        session_files_list.append({"filename": f["filename"], "samples": samples, "sr": sr})
    process_session_worker(task_id, session_files_list)

def drive_download_worker(task_id: str, folder_url: str):
    """Scrapes external drive storage via background worker thread."""
    download_dir = f"/tmp/drive_{task_id}"
    try:
        os.makedirs(download_dir, exist_ok=True)
        gdown.get_user_agent = lambda: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        gdown.download_folder(url=folder_url, output=download_dir, quiet=False, remaining_ok=True, use_cookies=False)
        
        raw_files = glob.glob(os.path.join(download_dir, "**/*"), recursive=True)
        valid_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')
        found_files = [f for f in raw_files if os.path.isfile(f) and f.lower().endswith(valid_extensions)]
        
        if not found_files:
            BACKGROUND_TASKS_DB[task_id] = {"success": False, "status": "failed", "detail": "Drive empty or inaccessible."}
            return
            
        def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        found_files.sort(key=natural_sort_key)
        
        session_files_list = []
        for path in found_files:
            try: samples, sr = sf.read(path)
            except Exception: samples, sr = librosa.load(path, sr=16000, mono=True)
            session_files_list.append({"filename": os.path.basename(path), "samples": samples, "sr": sr})
            
        process_session_worker(task_id, session_files_list)
    except Exception as e:
        BACKGROUND_TASKS_DB[task_id] = {"success": False, "status": "failed", "detail": f"Cloud Mapping Failure: {str(e)}"}
    finally:
        if os.path.exists(download_dir): shutil.rmtree(download_dir)

@api_router.get("/")
def serve_dashboard_interface():
    """Serves the decoupled HTML frontend."""
    # Resolves to the root directory where index.html is located
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")
    with open(html_path, "r", encoding="utf-8") as f: 
        html_body = f.read()
    return HTMLResponse(content=html_body, status_code=200)

@api_router.post("/analyze_session_batch/")
def analyze_session_batch(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    """Receives direct files and queues them into background AI pipelines."""
    task_id = str(uuid.uuid4().int)[:13]
    files_payload = [{"filename": f.filename, "bytes": f.file.read()} for f in files]
    BACKGROUND_TASKS_DB[task_id] = {"success": True, "status": "processing"}
    background_tasks.add_task(batch_upload_worker, task_id, files_payload)
    return {"success": True, "task_id": task_id}

@api_router.post("/analyze_google_drive/")
def analyze_google_drive(payload: DrivePayload, background_tasks: BackgroundTasks):
    """Receives drive link and queues the scraper process."""
    task_id = str(uuid.uuid4().int)[:13]
    BACKGROUND_TASKS_DB[task_id] = {"success": True, "status": "processing"}
    background_tasks.add_task(drive_download_worker, task_id, payload.drive_url)
    return {"success": True, "task_id": task_id}

@api_router.get("/check_task/{task_id}")
def check_task_status(task_id: str):
    """Polling endpoint for the frontend to check task resolution."""
    return BACKGROUND_TASKS_DB.get(task_id, {"status": "failed", "detail": "Task ID not found."})