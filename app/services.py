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

def chunk_audio(samples, sr, max_duration=30):
    """Slices long continuous audio into fixed 30-second maximum tensor windows."""
    chunk_length = max_duration * sr
    return [samples[i:i + chunk_length] for i in range(0, len(samples), chunk_length)]

def core_processing_pipeline(session_files_list: list):
    session_timeline = []
    t_start = time.time()
    clean_audios_list = []
    valid_metadata = []
    
    for file_index, file_item in enumerate(session_files_list):
        samples = file_item["samples"]
        sr = file_item["sr"]
        
        # Resample and denoise
        if samples.ndim > 1:
            samples = np.mean(samples, axis=0 if samples.shape[0] < samples.shape[1] else 1)
        if sr != 16000:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        samples = samples.astype(np.float32)
        samples = nr.reduce_noise(y=samples, sr=16000, prop_decrease=0.8)
        
        # Enforce 30-second max processing units
        chunks = chunk_audio(samples, 16000, max_duration=30)
        for c_idx, chunk in enumerate(chunks):
            clean_audios_list.append(np.ascontiguousarray(chunk, dtype=np.float32))
            valid_metadata.append({"filename": file_item["filename"], "file_index": file_index, "chunk_index": c_idx})
            
    if not clean_audios_list:
        return [], 0
            
    stream_gpu0 = torch.cuda.Stream(device=gpu_0["index"])
    stream_gpu1 = torch.cuda.Stream(device=gpu_1["index"])
    
    for step in range(0, len(clean_audios_list), BATCH_SIZE_LIMIT):
        batch_audios = clean_audios_list[step : step + BATCH_SIZE_LIMIT]
        batch_meta = valid_metadata[step : step + BATCH_SIZE_LIMIT]
        
        with torch.cuda.stream(stream_gpu0):
            wh_inputs = whisper_processor(batch_audios, sampling_rate=16000, return_attention_mask=True, return_tensors="pt", padding=True)
            input_features = wh_inputs.input_features.to(device=gpu_0["name"])
            asr_attention_mask = wh_inputs.attention_mask.to(device=gpu_0["name"])
            with torch.no_grad():
                predicted_batch_ids = whisper_model.generate(input_features, attention_mask=asr_attention_mask, max_new_tokens=150, num_beams=1)
                
        with torch.cuda.stream(stream_gpu1):
            audio_inputs = audio_processor(batch_audios, sampling_rate=16000, return_attention_mask=True, return_tensors="pt", padding=True)
            audio_inputs = {k: v.to(device=gpu_1["name"]) for k, v in audio_inputs.items()}
            with torch.no_grad():
                batch_logits = audio_model(**audio_inputs).logits
                predicted_batch_vocal_ids = torch.argmax(batch_logits, dim=-1).cpu().numpy()

        torch.cuda.synchronize(device=gpu_0["index"])
        torch.cuda.synchronize(device=gpu_1["index"])
        
        clean_transcripts = [t.strip() for t in whisper_processor.batch_decode(predicted_batch_ids, skip_special_tokens=True)]
        sentiment_results = sentiment_pipeline(clean_transcripts)
        
        for i, meta in enumerate(batch_meta):
            if not clean_transcripts[i]:
                continue
            session_timeline.append({
                "transcript": clean_transcripts[i],
                "semantic_sentiment": sentiment_results[i]["label"],
                "vocal_acoustic_emotion": audio_id2label.get(predicted_batch_vocal_ids[i], "Neutral"),
                "timestamp_sec": meta["chunk_index"] * 30
            })
            
    total_pipeline_latency = round(time.time() - t_start, 4)
    return session_timeline, total_pipeline_latency

def generate_clinical_report(session_timeline):
    t_llm_start = time.time()
    prompt_context = "".join(f"[{c['timestamp_sec']}s] Text: '{c['transcript']}' | Mood: {c['vocal_acoustic_emotion']}\n" for c in session_timeline)
    
    system_instruction = (
        "You are an expert psychiatric assistant. Write a clinical report in English analyzing the patient's emotional journey. "
        "Format strictly with Markdown headers: ### 1️⃣ Psychodynamic Summary, ### 2️⃣ Energy Curve, ### 3️⃣ Recommendations."
    )
    full_prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\nData:\n{prompt_context}<|im_end|>\n<|im_start|>assistant\n"
    
    llm_res = qwen_pipeline(full_prompt, max_new_tokens=1024, temperature=0.3, repetition_penalty=1.15)
    report = llm_res[0]["generated_text"].split("<|im_start|>assistant\n")[-1]
    
    html_report = report.replace("### 1️⃣", "<h3 class='gradient-text'>1️⃣").replace("### 2️⃣", "<h3 class='gradient-text'>2️⃣").replace("### 3️⃣", "<h3 class='gradient-text'>3️⃣")
    llm_latency = round(time.time() - t_llm_start, 4)
    
    return html_report.strip(), llm_latency
