"""Check that the container environment matches what the Qwen3-TTS repo needs.

Usage:  python docker/verify_env.py            # deps / GPU / tools
        python docker/verify_env.py --smoke    # + end-to-end TTS with the 0.6B model
"""
import argparse
import importlib
import importlib.metadata as md
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def parse_req(req):
    for op in ("==", ">=", "<="):
        if op in req:
            n, v = req.split(op, 1)
            return n.strip(), op, v.strip()
    return req.strip(), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("== Python")
    check("python >= 3.9 (README recommends 3.12)", sys.version_info >= (3, 9), sys.version.split()[0])

    print("== pyproject.toml dependencies")
    proj = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    for req in proj["dependencies"]:
        name, op, want = parse_req(req)
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            check(name, False, "not installed")
            continue
        check(name, op != "==" or have == want, f"{have}" + (f" (required {want})" if op else ""))

    print("== qwen-tts package")
    import qwen_tts
    check("qwen_tts imports from repo", Path(qwen_tts.__file__).resolve().is_relative_to(REPO), qwen_tts.__file__)
    for mod in ["qwen_tts.inference.qwen3_tts_model", "qwen_tts.inference.qwen3_tts_tokenizer",
                "librosa", "sox", "onnxruntime", "einops", "gradio", "torchaudio", "soundfile"]:
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except Exception as e:  # noqa: BLE001
            check(f"import {mod}", False, repr(e))

    print("== CUDA / GPU")
    import torch
    check("torch CUDA available", torch.cuda.is_available(),
          f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"     {p.name}, sm_{p.major}{p.minor}, free {free/2**30:.1f} / {total/2**30:.1f} GiB")
        check("bf16 supported", torch.cuda.is_bf16_supported())
        try:
            from flash_attn import flash_attn_func, __version__ as fa_v
            q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
            out = flash_attn_func(q, q, q, causal=True)
            ref = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2), is_causal=True).transpose(1, 2)
            check("flash-attn 2 kernel runs & matches SDPA", (out - ref).abs().max().item() < 1e-2, fa_v)
        except Exception as e:  # noqa: BLE001
            check("flash-attn 2", False, repr(e))

    print("== Tools")
    for tool in ["tmux", "git", "sox", "ffmpeg", "nvcc", "nsys", "ncu", "nvidia-smi", "claude", "py-spy", "htop", "nvtop"]:
        path = shutil.which(tool)
        check(tool, path, path or "not found")
    if shutil.which("nvcc"):
        print("     " + subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1])

    if args.smoke:
        print("== Smoke test: Qwen3-TTS-12Hz-0.6B-CustomVoice")
        import time
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device_map="cuda:0",
            dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        speakers = model.get_supported_speakers()
        t0 = time.time()
        wavs, sr = model.generate_custom_voice(
            text="其实我真的有发现，我是一个特别善于观察别人情绪的人。", language="Chinese", speaker=speakers[0])
        dt = time.time() - t0
        out = Path("/tmp/verify_smoke.wav")
        sf.write(out, wavs[0], sr)
        dur = len(wavs[0]) / sr
        check("generate_custom_voice", dur > 0.5,
              f"{dur:.2f}s audio @ {sr}Hz in {dt:.2f}s (RTF {dt/dur:.2f}), speaker={speakers[0]} -> {out}")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
