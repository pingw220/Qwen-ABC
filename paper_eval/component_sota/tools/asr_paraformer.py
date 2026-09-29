"""Mandarin ASR (FunASR Paraformer-zh, iic/speech_seaco_paraformer_large...) for a list of wavs (conda env midi-sag).
Writes <wav>.asr.json {text}. The same recognizer is used for every renderer and condition.
fsmn-vad segments the song first: without VAD Paraformer transcribes only a fragment of a multi-minute
song (11 of 350 characters on a test song); earlier no-VAD outputs were kept as <wav>.asr_novad.json."""
import json, os, sys
os.environ.setdefault("MODELSCOPE_CACHE", "/gscratch/ark/pingw220/third_party/modelscope")
os.environ["PATH"] = "/gscratch/ark/pingw220/miniconda3/envs/aria_amt/bin:" + os.environ["PATH"]   # an ffmpeg binary
from funasr import AutoModel
m = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000}, disable_update=True, device="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
for p in open(sys.argv[1]).read().split():
    res = m.generate(input=p, batch_size_s=300)
    text = "".join(r.get("text", "") for r in res).replace(" ", "")
    json.dump({"text": text, "vad": "fsmn-vad"}, open(p + ".asr.json", "w"), ensure_ascii=False)
    print("asr", p, len(text), flush=True)
