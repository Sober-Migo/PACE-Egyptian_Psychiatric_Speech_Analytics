# Core AI services, audio processing, and LLM report generation
import time
import numpy as np
import librosa
import noisereduce as nr
import torch
from app.ml_models import (
    gpu_0, gpu_1, whisper_processor, whisper_model, 
    sentiment_pipeline, audio_processor, audio_model, 
    audio_id2label, qwen_pipeline
)

BATCH_SIZE_LIMIT = 8

def preprocess_incoming_audio_buffer(samples, sr) -> np.ndarray:
    """Resamples and removes stationary noise from clinical recordings."""
    try:
        if samples.ndim > 1:
            samples = np.mean(samples, axis=0 if samples.shape[0] < samples.shape[1] else 1)
        if sr != 16000:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        samples = samples.astype(np.float32)
        samples = nr.reduce_noise(y=samples, sr=16000, prop_decrease=0.8)
        return samples
    except Exception as e:
        print(f"🚨 Preprocessing Signal Failure: {e}")
        return None

def core_processing_pipeline(session_files_list: list):
    """Executes the dual-stream multimodal AI pipeline across isolated GPUs."""
    session_timeline = []
    t_start = time.time()
    clean_audios_list = []
    valid_metadata = []
    
    for index, file_item in enumerate(session_files_list):
        clean_speech = preprocess_incoming_audio_buffer(file_item["samples"], file_item["sr"])
        if clean_speech is not None and clean_speech.size > 0:
            clean_audios_list.append(np.ascontiguousarray(clean_speech, dtype=np.float32))
            valid_metadata.append({"filename": file_item["filename"], "index": index})
            
    if not clean_audios_list:
        return [], 0
        
    stream_gpu0 = torch.cuda.Stream(device=gpu_0["index"])
    stream_gpu1 = torch.cuda.Stream(device=gpu_1["index"])
    total_elements = len(clean_audios_list)
    
    for step in range(0, total_elements, BATCH_SIZE_LIMIT):
        batch_audios = clean_audios_list[step : step + BATCH_SIZE_LIMIT]
        batch_meta = valid_metadata[step : step + BATCH_SIZE_LIMIT]
        
        # GPU 0: Whisper Execution
        t_g0 = time.time()
        with torch.cuda.stream(stream_gpu0):
            wh_inputs = whisper_processor(batch_audios, sampling_rate=16000, return_attention_mask=True, return_tensors="pt", padding=True)
            input_features = wh_inputs.input_features.to(device=gpu_0["name"])
            asr_attention_mask = wh_inputs.attention_mask.to(device=gpu_0["name"])
            with torch.no_grad():
                predicted_batch_ids = whisper_model.generate(input_features, attention_mask=asr_attention_mask, max_new_tokens=150, num_beams=1)
                
        # GPU 1: Acoustic Emotion Execution
        t_g1 = time.time()
        with torch.cuda.stream(stream_gpu1):
            audio_inputs = audio_processor(batch_audios, sampling_rate=16000, return_attention_mask=True, return_tensors="pt", padding=True)
            audio_inputs = {k: v.to(device=gpu_1["name"]) for k, v in audio_inputs.items()}
            with torch.no_grad():
                batch_logits = audio_model(**audio_inputs).logits
                predicted_batch_vocal_ids = torch.argmax(batch_logits, dim=-1).cpu().numpy()

        # Synchronize streams before decoding
        torch.cuda.synchronize(device=gpu_0["index"])
        torch.cuda.synchronize(device=gpu_1["index"])
        
        # CPU Post-processing
        transcripts_list = whisper_processor.batch_decode(predicted_batch_ids, skip_special_tokens=True)
        clean_transcripts = [t.strip() for t in transcripts_list]
        sentiment_results = sentiment_pipeline(clean_transcripts)
        
        for i, meta in enumerate(batch_meta):
            if not clean_transcripts[i]:
                continue
            vocal_emotion = audio_id2label.get(predicted_batch_vocal_ids[i], f"Class_{predicted_batch_vocal_ids[i]}")
            session_timeline.append({
                "transcript": clean_transcripts[i],
                "semantic_sentiment": sentiment_results[i]["label"],
                "vocal_acoustic_emotion": vocal_emotion,
            })
            
    total_pipeline_latency = round(time.time() - t_start, 4)
    return session_timeline, total_pipeline_latency

def generate_clinical_report(session_timeline):
    """Generates the structured clinical Markdown report via the Qwen LLM."""
    t_llm_start = time.time()
    timeline_summary_prompt = "".join(f" Transcript: '{c['transcript']}' | Semantic Context: {c['semantic_sentiment']} | Vocal Energy: {c['vocal_acoustic_emotion']}\n" for c in session_timeline)
    
    system_instruction = (
        "You are an expert psychiatric assistant specializing in analyzing psychotherapy session transcripts and behavioral acoustic indicators. "
        "Your task is to write a comprehensive and organized psychological analysis report based on the provided temporal session data. "
        "The patients are speaking in Egyptian Arabic. You must format the report using exactly the following Markdown headers:\n\n"
        "### 1️⃣ General Psychodynamic Summary\n\n"
        "### 2️⃣ Mood and Acoustic Energy Curve Analysis\n\n"
        "### 3️⃣ Clinical Notes and Therapist Recommendations"
    )
    full_prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\nAnalyze this specific session data for immediate clinical review:\n\n{timeline_summary_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    llm_res = qwen_pipeline(full_prompt, max_new_tokens=2560, temperature=0.3, do_sample=True, repetition_penalty=1.15)
    clinical_insight_report = llm_res[0]["generated_text"].split("<|im_start|>assistant\n")[-1]
    
    html_report = clinical_insight_report.replace("### 1️⃣", "<h3 style='color:#1e3a8a;'>1️⃣").replace("### 2️⃣", "<h3 style='color:#1e3a8a;'>2️⃣").replace("### 3️⃣", "<h3 style='color:#1e3a8a;'>3️⃣")
    llm_latency = round(time.time() - t_llm_start, 4)
    
    return html_report.strip(), llm_latency