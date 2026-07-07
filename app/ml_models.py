import torch
from transformers import pipeline, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
from app.config import WHISPER_MODEL_PATH, BAVED_MODEL_PATH, QWEN_MODEL_PATH, CAMELBERT_MODEL_PATH

gpu_0 = {"name": "cuda:0", "index": 0}
has_multi_gpu = torch.cuda.device_count() > 1
gpu_1 = {"name": "cuda:1" if has_multi_gpu else "cuda:0", "index": 1 if has_multi_gpu else 0}
print(f"🎮 [BALANCER] Loaded asymmetric hardware -> GPU_0: {gpu_0} | GPU_1: {gpu_1}")

print("📦 Loading GPU_0 Models (ASR Engine + Sentiment)...")
whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH, torch_dtype=torch.float32).to(gpu_0["name"])
whisper_model.eval()

whisper_model.generation_config.language = "arabic"
whisper_model.generation_config.task = "transcribe"
whisper_model.generation_config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
whisper_model.generation_config.max_length = 448 

sentiment_pipeline = pipeline("text-classification", model=CAMELBERT_MODEL_PATH, device=gpu_0["index"])

print("📦 Loading GPU_1 Models (LLM + Acoustic AER)...")
qwen_pipeline = pipeline("text-generation", model=QWEN_MODEL_PATH, dtype=torch.float16, device=gpu_1["index"])
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(BAVED_MODEL_PATH)
audio_model = AutoModelForAudioClassification.from_pretrained(BAVED_MODEL_PATH, dtype=torch.float32).to(gpu_1["name"])
audio_model.eval()
audio_id2label = audio_model.config.id2label

print("🏁 [SUCCESS] All target core networks synchronized on multi-GPU pipeline layout.")
