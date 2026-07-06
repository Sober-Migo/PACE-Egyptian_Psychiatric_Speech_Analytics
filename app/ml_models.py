# Multi-GPU hardware allocation and model initialization
import torch
from transformers import pipeline, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
from app.config import authenticate_huggingface

# Trigger authentication before loading architectures
authenticate_huggingface()

# 1. Hardware Allocation Mapping
gpu_0 = {"name": "cuda:0", "index": 0}
has_multi_gpu = torch.cuda.device_count() > 1
gpu_1 = {"name": "cuda:1" if has_multi_gpu else "cuda:0", "index": 1 if has_multi_gpu else 0}
print(f"🎮 [BALANCER] Loaded asymmetric hardware -> GPU_0: {gpu_0} | GPU_1: {gpu_1}")

# 2. GPU 0 Models: ASR & Sentiment
print("📦 Loading GPU_0 Models (ASR Engine + Sentiment Classifier)...")
WHISPER_MODEL_ID = "am4magdy/egyptian-whisper-large-v3-standalone"
whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID, torch_dtype=torch.float32).to(gpu_0["name"])
whisper_model.eval()

# Native locks inside generation configuration config layout to freeze language targets
whisper_model.generation_config.language = "arabic"
whisper_model.generation_config.task = "transcribe"
whisper_model.generation_config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
whisper_model.generation_config.max_length = 448 

sentiment_pipeline = pipeline("text-classification", model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment", device=gpu_0["index"])

# 3. GPU 1 Models: LLM & Acoustic AER
print("📦 Loading GPU_1 Models (LLM Report Generator + Vocal Acoustic Classifier)...")
qwen_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16, device=gpu_1["index"])

BAVED_MODEL_ID = "am4magdy/Baved_3e5b16"
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(BAVED_MODEL_ID)
audio_model = AutoModelForAudioClassification.from_pretrained(BAVED_MODEL_ID, dtype=torch.float32).to(gpu_1["name"])
audio_model.eval()
audio_id2label = audio_model.config.id2label

print("🏁 [SUCCESS] All target core networks synchronized on multi-GPU pipeline layout.")