# -*- coding: utf-8 -*-
"""
Sweep caller for sd_unet_psy_acts.py.
- Runs the original generator multiple times (serially) with different knobs.
- Each run writes: <OUT>_A_baseline.png and <OUT>_B_patched.png
- This script collects metrics vs each run's baseline, writes results.csv, and
  produces a contact sheet (thumbnails labeled with settings + LPIPS).

Example:
  python sweep_call_psy_generator.py \
    --prompt "a cozy reading nook with warm ambient light, 35mm film" \
    --acts silu,mish,gelu_tanh \
    --taus 1.0,1.2,1.4 \
    --betas 0.0,0.15,0.3 \
    --gamma 1.0 \
    --stages up,mid \
    --start-idx 0 \
    --steps 30 \
    --cfg 7.5 \
    --seed 12345 \
    --outdir sweep_out \
    --calibrate \
    --patch-mlp
"""
import argparse, csv, itertools, os, math, subprocess, sys, shlex, traceback
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Optional metrics
try:
    import lpips as _lpips
    LPIPS_OK = True
except Exception:
    _lpips, LPIPS_OK = None, False

try:
    import cv2 as _cv2
    CV2_OK = True
except Exception:
    _cv2, CV2_OK = None, False


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run_generator(gen_path: str, args_list: List[str]) -> int:
    """
    Calls: python gen_path <args_list>
    Returns process returncode.
    """
    cmd = [sys.executable, gen_path] + args_list
    print("[CALL]", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)

def pil_open(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im

def mse(imgA: Image.Image, imgB: Image.Image) -> float:
    a = np.asarray(imgA).astype(np.float32)
    b = np.asarray(imgB).astype(np.float32)
    return float(((a - b) ** 2).mean())

def entropy(img: Image.Image) -> float:
    gray = img.convert("L")
    hist = np.array(gray.histogram(), dtype=np.float64)
    p = hist / max(hist.sum(), 1e-12)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())

def sharpness_var_laplacian(img: Image.Image) -> float:
    if not CV2_OK:
        return float("nan")
    gray = np.array(img.convert("L"))
    lap = _cv2.Laplacian(gray, _cv2.CV_64F)
    return float(lap.var())

def lpips_score(imgA: Image.Image, imgB: Image.Image) -> float:
    if not LPIPS_OK:
        return float("nan")
    # LPIPS expects [-1,1] normalized tensors
    import torch
    to_t = lambda im: (torch.from_numpy(np.asarray(im).astype(np.float32)/255.0)
                       .permute(2,0,1).unsqueeze(0)*2.0 - 1.0)
    dev = "cuda" if _has_cuda() else "cpu"
    loss_fn = _lpips.LPIPS(net='alex').to(dev)
    with torch.no_grad():
        d = loss_fn(to_t(imgA).to(dev), to_t(imgB).to(dev))
    return float(d.squeeze().detach().cpu().numpy())

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def make_contact_sheet(items, thumb=384, cols=3, label=True, font=None, save_path="contact_sheet.png"):
    """
    items: List[Tuple[PIL.Image, caption:str]]
    """
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
    rows = math.ceil(len(items)/cols)
    pad = 8
    caption_h = 52 if label else 0
    w = cols*thumb + (cols+1)*pad
    h = rows*(thumb + caption_h) + (rows+1)*pad
    sheet = Image.new("RGB", (w, h), (30,30,30))
    draw  = ImageDraw.Draw(sheet)

    for idx, (im, cap) in enumerate(items):
        r = idx // cols
        c = idx % cols
        x0 = pad + c*(thumb) + c*pad
        y0 = pad + r*(thumb + caption_h) + r*pad
        imt = im.copy().resize((thumb, thumb), Image.LANCZOS)
        sheet.paste(imt, (x0, y0))
        if label:
            draw.multiline_text((x0, y0+thumb+4), cap, fill=(230,230,230), font=font, spacing=2)
    sheet.save(save_path)
    return sheet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-path", type=str, default="sd_unet_psy_acts.py",
                    help="Path to the original generator script")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--acts", type=str, default="silu,mish,gelu_tanh")
    ap.add_argument("--taus", type=str, default="1.0,1.2,1.4")
    ap.add_argument("--betas", type=str, default="0.0,0.15,0.3")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--stages", type=str, default="up,mid")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--patch-mlp", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", type=str, default="sweep_out")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--subprocess", action="store_true",
                    help="Force subprocess execution per run (disables pipeline reuse).")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    acts  = [s.strip() for s in args.acts.split(",") if s.strip()]
    taus  = [float(x) for x in args.taus.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    # Contact sheet items (start with a single baseline from the first combo)
    thumbs = []
    rows = []
    header = ["act","tau","beta","gamma","stages","start_idx","patch_mlp",
              "lpips","mse","entropy","sharpness","patched_path","baseline_path","out_tag"]
    rows.append(header)

    combolist = list(itertools.product(acts, taus, betas))

    reuse = not args.subprocess
    pipe = None
    run_kwargs = None
    if reuse:
        import sd_unet_psy_acts as psy
        pipe = psy.load_pipeline()
        run_kwargs = {
            "prompt": args.prompt,
            "steps": args.steps,
            "cfg": args.cfg,
            "seed": args.seed,
            "gamma": args.gamma,
            "stages": ",".join(stages),
            "start_idx": args.start_idx,
            "patch_mlp": args.patch_mlp,
            "calibrate": args.calibrate,
        }

    # Generate once per combo by calling the original generator
    for idx, (act, tau, beta) in enumerate(combolist, start=1):
        tag = f"act={act}_tau={tau}_beta={beta}_gamma={args.gamma}_stg={'-'.join(stages)}_idx={args.start_idx}{'_mlp' if args.patch_mlp else ''}"
        out_base = os.path.join(args.outdir, f"{tag}.png")

        gen_args = [
            "--prompt", args.prompt,
            "--out", out_base,
            "--steps", str(args.steps),
            "--cfg", str(args.cfg),
            "--seed", str(args.seed),
            "--act", act,
            "--tau", str(tau),
            "--beta", str(beta),
            "--gamma", str(args.gamma),
            "--stages", ",".join(stages),
            "--start-idx", str(args.start_idx)
        ]
        if args.patch_mlp: gen_args.append("--patch-mlp")
        if args.calibrate: gen_args.append("--calibrate")

        if reuse:
            try:
                # per-combo overrides
                per_run = dict(run_kwargs)
                per_run.update({
                    "out": out_base,
                    "act": act,
                    "tau": tau,
                    "beta": beta,
                })
                psy.run_from_kwargs(pipe=pipe, **per_run)
                rc = 0
            except Exception as exc:
                print(f"[WARN] Generator exception for {tag}: {exc}")
                traceback.print_exc()
                rc = 1
        else:
            rc = run_generator(args.gen_path, gen_args)

        if rc != 0:
            print(f"[WARN] Generator returned {rc} for {tag}; skipping metrics.")
            continue

        base_path   = out_base.replace(".png", "_A_baseline.png")
        patch_path  = out_base.replace(".png", "_B_patched.png")
        if not (os.path.exists(base_path) and os.path.exists(patch_path)):
            print(f"[WARN] Missing outputs for {tag}; expected {base_path} and {patch_path}")
            continue

        base_img  = pil_open(base_path)
        patch_img = pil_open(patch_path)

        # Collect baseline only once for the sheet (first run)
        if idx == 1:
            thumbs.append((base_img, "baseline"))

        # Metrics vs THIS run's own baseline
        lp  = lpips_score(base_img, patch_img)
        m   = mse(base_img, patch_img)
        ent = entropy(patch_img)
        shp = sharpness_var_laplacian(patch_img)

        cap = f"{act}\nτ={tau} β={beta}\nLPIPS={lp if not np.isnan(lp) else float('nan'):.3f}"
        thumbs.append((patch_img, cap))

        rows.append([act, tau, beta, args.gamma, "-".join(stages), args.start_idx, int(args.patch_mlp),
                     lp, m, ent, shp, patch_path, base_path, tag])

    # Save CSV
    csv_path = os.path.join(args.outdir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(rows)
    print("[SAVE] metrics ->", csv_path)

    # Contact sheet
    cs_path = os.path.join(args.outdir, "contact_sheet.png")
    make_contact_sheet(thumbs, thumb=384, cols=args.cols, label=True, save_path=cs_path)
    print("[SAVE] contact sheet ->", cs_path)
    print("[DONE]")

if __name__ == "__main__":
    main()
