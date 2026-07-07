import os
import tempfile
import librosa
import soundfile as sf
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import HTMLResponse
from app.services import core_processing_pipeline, generate_clinical_report

api_router = APIRouter()

@api_router.get("/")
def serve_dashboard_interface():
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")
    with open(html_path, "r", encoding="utf-8") as f: 
        html_body = f.read()
    return HTMLResponse(content=html_body, status_code=200)

@api_router.post("/analyze/")
def analyze_session(files: list[UploadFile] = File(...)):
    session_files = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(f.file.read())
            tmp_p = tmp.name
        try:
            samples, sr = sf.read(tmp_p)
        except:
            samples, sr = librosa.load(tmp_p, sr=16000, mono=True)
        finally:
            if os.path.exists(tmp_p): os.remove(tmp_p)
        session_files.append({"filename": f.filename, "samples": samples, "sr": sr})
        
    timeline, pipeline_latency = core_processing_pipeline(session_files)
    report, llm_latency = generate_clinical_report(timeline)
    
    return {
        "timeline": timeline, 
        "report": report,
        "latency": {"pipeline": pipeline_latency, "llm": llm_latency}
    }
