import sys, numpy as np, wave
from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=2)
for f in sys.argv[1:]:
    w=wave.open(f); a=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
    for temp in [0.0]:
        segs,_=m.transcribe(a, language="pt", beam_size=5, initial_prompt="Vídeo para testar o Claude. Coloco uma roupa em mim.")
        print(f, [s.text for s in segs])
        segs,_=m.transcribe(a, language="pt", beam_size=5)
        print(f, [s.text for s in segs])
