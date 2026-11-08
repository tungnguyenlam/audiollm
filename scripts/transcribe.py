"""Transcribe one audio file using a local or downloaded AudioLLM backup."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--step", help="retained step number, or latest for checkpoint/")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--hypothesis", help="enable ASR correction with this hypothesis")
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    from audiollm import config as C
    from audiollm.features import LogMelFeaturizer
    from audiollm.model import SpeechLLM
    from audiollm.utils import compute_context

    run = args.run.resolve()
    config = C.with_legacy_defaults(json.loads((run / "config.json").read_text()))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config.update(DEVICE=device, DTYPE="float32" if device == "cpu" else "bfloat16",
                  PRECISION="fp32" if device == "cpu" else "bf16", GRAD_CKPT=False)
    C.apply_config(config)
    weights = run if args.step is None else (run / "checkpoint" if args.step == "latest" else
                                            run / "checkpoints" / f"step-{int(args.step):09d}")
    if not (weights / "proj.pt").exists():
        parser.error(f"no weights at {weights}; choose --step latest or a retained step")
    model = SpeechLLM.load(weights).eval()
    features = LogMelFeaturizer(torch.load(run / "mvn.pt", map_location="cpu", weights_only=True))
    waveform, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if rate != C.SR:
        waveform = soxr.resample(waveform, rate, C.SR)
    if len(waveform) < C.N_FFT:
        parser.error("audio is too short")
    mel = features.from_waveform(np.asarray(waveform, dtype=np.float32))
    prompt = None if args.hypothesis is None else model.tokenizer(
        C.CORRECTION_PROMPT.format(hyp=" " + args.hypothesis), add_special_tokens=False).input_ids
    batch = dict(features=mel[None], feature_lengths=torch.tensor([len(mel)]), target=[[]],
                 sec=[len(waveform) / C.SR], prompt_ids=[prompt])
    with torch.inference_mode(), compute_context():
        transcripts, _ = model.transcribe(batch)
    print(transcripts[0])


if __name__ == "__main__":
    main()
