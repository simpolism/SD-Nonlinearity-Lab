"""
Local Flask web UI for sd_unet_psy_acts.

Provides a simple 90s-psychedelic themed control panel to tweak activation knobs
and preview baseline vs patched outputs without running a full sweep.
"""

from __future__ import annotations

import hashlib
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for
import torch

import sd_unet_psy_acts as psy

# ---------------------- Configuration ----------------------

APP = Flask(__name__)
APP.config["TEMPLATES_AUTO_RELOAD"] = True

OUTPUT_DIR = Path("web_out")
BASELINE_CACHE_DIR = OUTPUT_DIR / "baseline_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CHOICES = [
    {"id": "runwayml/stable-diffusion-v1-5", "label": "SD v1.5 (Base)", "pipeline": "auto", "dtype": "fp16"},
    {"id": "stabilityai/stable-diffusion-2-1-base", "label": "SD v2.1 Base", "pipeline": "auto", "dtype": "fp16"},
    {"id": "stabilityai/stable-diffusion-xl-base-1.0", "label": "SDXL Base 1.0", "pipeline": "sdxl", "dtype": "fp16"},
    {"id": "stabilityai/stable-diffusion-xl-refiner-1.0", "label": "SDXL Refiner 1.0", "pipeline": "sdxl", "dtype": "fp16"},
    {"id": "stabilityai/stable-diffusion-3-medium", "label": "SD 3 Medium", "pipeline": "sd3", "dtype": "fp16"},
]

DEFAULT_OPTS = {
    "prompt": "a cozy reading nook with warm ambient light, 35mm film grain",
    "negative_prompt": "",
    "steps": 25,
    "cfg": 7.5,
    "seed": 12345,
    "act": "silu",
    "tau": 1.0,
    "beta": 0.0,
    "gamma": 1.0,
    "stages": ["up", "mid"],
    "start_idx": 0,
    "start_idx_down": 0,
    "start_idx_mid": 0,
    "start_idx_up": 0,
    "end_idx": None,
    "end_idx_down": None,
    "end_idx_mid": None,
    "end_idx_up": None,
    "patch_mlp": False,
    "calibrate": True,
    "height": 512,
    "width": 512,
    "model_id": MODEL_CHOICES[0]["id"],
    "pipeline": MODEL_CHOICES[0]["pipeline"],
    "dtype": MODEL_CHOICES[0]["dtype"],
    "skip_baseline": False,
}

PIPELINE_CACHE: Dict[Tuple[str, str, str], Any] = {}
PIPELINE_LOCK = threading.Lock()
BASELINE_CACHE: Dict[Tuple, psy.BaselineRecord] = {}

STAGE_ORDER = ["down", "mid", "up"]
STAGE_SLOT_COUNT = 8  # approximate depth slots per stage for visualization
STAGE_TITLES = {
    "down": "Down",
    "mid": "Mid",
    "up": "Up",
}
DEFAULT_STAGE_DEPTHS = {stage: STAGE_SLOT_COUNT for stage in STAGE_ORDER}

# ---------------------- Helpers ----------------------


def _offload_inactive_pipelines(active_key: Tuple[str, str, str]):
    if not torch.cuda.is_available():
        return
    for key, pipe in list(PIPELINE_CACHE.items()):
        if key == active_key:
            continue
        try:
            pipe.to("cpu")
        except Exception:
            continue
    torch.cuda.empty_cache()


def _ensure_pipeline_device(pipe):
    target = psy.device
    try:
        pipe.to(target)
    except Exception:
        pass


def get_pipeline(model_id: str, pipeline_kind: str, dtype: str):
    key = (model_id, pipeline_kind, dtype)
    if key not in PIPELINE_CACHE:
        with PIPELINE_LOCK:
            if key not in PIPELINE_CACHE:
                PIPELINE_CACHE[key] = psy.load_pipeline(
                    model_id=model_id,
                    pipeline_kind=pipeline_kind,
                    dtype=psy.resolve_dtype(dtype),
                    device_override=psy.device,
                    attention_slicing=True,
                    vae_tiling=False,
                    vae_slicing=False,
                    cpu_offload=False,
                )
    pipe = PIPELINE_CACHE[key]
    _offload_inactive_pipelines(key)
    _ensure_pipeline_device(pipe)
    return pipe


def _slugify(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)[:48]


def _fmt_float(val: float) -> str:
    return f"{val:.3f}".rstrip("0").rstrip(".")


def _build_baseline_key(opts: dict) -> Tuple:
    key_fields = (
        opts["prompt"].strip(),
        opts.get("negative_prompt", "").strip(),
        opts["steps"],
        opts["cfg"],
        opts["seed"],
        opts["act"],
        float(opts["tau"]),
        float(opts["beta"]),
        float(opts["gamma"]),
        bool(opts["patch_mlp"]),
        bool(opts["calibrate"]),
        opts.get("height"),
        opts.get("width"),
        opts.get("model_id"),
        opts.get("pipeline"),
        opts.get("dtype"),
    )
    return key_fields


def _baseline_path_for_key(key: Tuple) -> Path:
    h = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
    return BASELINE_CACHE_DIR / f"baseline_{h}.png"


def _run_generation(form):
    opts = {
        "prompt": form.get("prompt", DEFAULT_OPTS["prompt"]).strip(),
        "negative_prompt": form.get("negative_prompt", "").strip(),
        "steps": int(form.get("steps", DEFAULT_OPTS["steps"])),
        "cfg": float(form.get("cfg", DEFAULT_OPTS["cfg"])),
        "seed": int(form.get("seed", DEFAULT_OPTS["seed"])),
        "act": form.get("act", DEFAULT_OPTS["act"]),
        "tau": float(form.get("tau", DEFAULT_OPTS["tau"])),
        "beta": float(form.get("beta", DEFAULT_OPTS["beta"])),
        "gamma": float(form.get("gamma", DEFAULT_OPTS["gamma"])),
        "stages": form.getlist("stages") or DEFAULT_OPTS["stages"],
        "patch_mlp": form.get("patch_mlp") == "on",
        "calibrate": form.get("calibrate") == "on",
        "height": int(form.get("height", DEFAULT_OPTS["height"])) if form.get("height") else None,
        "width": int(form.get("width", DEFAULT_OPTS["width"])) if form.get("width") else None,
        "skip_baseline": form.get("skip_baseline") == "on",
    }

    raw_step_start = form.get("step_start")
    try:
        opts["step_start"] = max(0, int(raw_step_start)) if raw_step_start not in (None, "") else 0
    except ValueError:
        opts["step_start"] = 0
    raw_step_end = form.get("step_end")
    if raw_step_end in (None, ""):
        step_end_val: Optional[int] = None
    else:
        try:
            step_end_val = max(opts["step_start"], int(raw_step_end))
        except ValueError:
            step_end_val = None
    opts["step_end"] = step_end_val

    sel_model_id = form.get("model_id", DEFAULT_OPTS["model_id"])
    model_info = next((m for m in MODEL_CHOICES if m["id"] == sel_model_id), MODEL_CHOICES[0])
    opts["model_id"] = model_info["id"]
    opts["pipeline"] = model_info["pipeline"]
    opts["dtype"] = model_info["dtype"]

    start_idx_map: Dict[str, int] = {}
    end_idx_map: Dict[str, Optional[int]] = {}
    for stage in STAGE_ORDER:
        field = f"start_idx_{stage}"
        hidden_field = f"{field}_hidden"
        raw_val = form.get(field)
        if raw_val is None:
            raw_val = form.get(hidden_field)
        default_val = DEFAULT_OPTS.get(field, 0)
        try:
            parsed = int(raw_val) if raw_val is not None else default_val
        except (TypeError, ValueError):
            parsed = default_val
        parsed = max(0, parsed)
        opts[field] = parsed
        start_idx_map[stage] = parsed

        end_field = f"end_idx_{stage}"
        end_hidden = f"{end_field}_hidden"
        raw_end = form.get(end_field)
        if raw_end is None:
            raw_end = form.get(end_hidden)
        default_end = DEFAULT_OPTS.get(end_field)
        if raw_end is None or raw_end == "":
            parsed_end = default_end
        else:
            try:
                parsed_end = max(0, int(raw_end))
            except (TypeError, ValueError):
                parsed_end = default_end
        if parsed_end is not None and parsed_end < parsed:
            parsed_end = parsed
        opts[end_field] = parsed_end
        end_idx_map[stage] = parsed_end

    opts["start_idx"] = min(start_idx_map.values()) if start_idx_map else 0
    finite_ends = [val for val in end_idx_map.values() if val is not None]
    opts["end_idx"] = max(finite_ends) if finite_ends else None

    stages_str = ",".join(opts["stages"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _slugify(opts["prompt"][:24] or "prompt")
    file_tag = f"{opts['act']}_tau{_fmt_float(opts['tau'])}_beta{_fmt_float(opts['beta'])}_gamma{_fmt_float(opts['gamma'])}_{stages_str}_{slug}_{timestamp}"
    out_path = OUTPUT_DIR / f"{file_tag}.png"

    baseline_key = None
    baseline_path = None
    cache_ref = None
    save_baseline_flag = False
    if not opts["skip_baseline"]:
        baseline_key = _build_baseline_key(opts)
        baseline_path = _baseline_path_for_key(baseline_key)
        cache_ref = BASELINE_CACHE
        save_baseline_flag = True

    pipe = get_pipeline(opts["model_id"], opts["pipeline"], opts["dtype"])
    baseline_path_str = str(baseline_path) if baseline_path is not None else None
    result = psy.run_from_kwargs(
        pipe=pipe,
        baseline_cache=cache_ref,
        baseline_key=baseline_key,
        baseline_path=baseline_path_str,
        save_baseline=save_baseline_flag,
        prompt=opts["prompt"],
        negative_prompt=opts["negative_prompt"] or None,
        out=str(out_path),
        steps=opts["steps"],
        cfg=opts["cfg"],
        seed=opts["seed"],
        act=opts["act"],
        tau=opts["tau"],
        beta=opts["beta"],
        gamma=opts["gamma"],
        stages=stages_str,
        start_idx=opts["start_idx"],
        start_idx_down=start_idx_map.get("down", opts["start_idx"]),
        start_idx_mid=start_idx_map.get("mid", opts["start_idx"]),
        start_idx_up=start_idx_map.get("up", opts["start_idx"]),
        end_idx=opts["end_idx"],
        end_idx_down=end_idx_map.get("down"),
        end_idx_mid=end_idx_map.get("mid"),
        end_idx_up=end_idx_map.get("up"),
        step_start=opts["step_start"],
        step_end=opts["step_end"],
        patch_mlp=opts["patch_mlp"],
        calibrate=opts["calibrate"],
        height=opts["height"],
        width=opts["width"],
        model_id=opts["model_id"],
        pipeline=opts["pipeline"],
        dtype=opts["dtype"],
        skip_baseline=opts["skip_baseline"],
    )

    stage_depths_raw = result.get("stage_depths") or {}
    stage_depths = {stage: int(stage_depths_raw.get(stage, DEFAULT_STAGE_DEPTHS.get(stage, STAGE_SLOT_COUNT)))
                    for stage in STAGE_ORDER}
    start_idx_result = result.get("start_idx_map") or {}
    start_idx_display = {stage: int(start_idx_result.get(stage, start_idx_map.get(stage, 0)))
                         for stage in STAGE_ORDER}
    end_idx_result = result.get("end_idx_map") or {}
    end_idx_display = {
        stage: end_idx_result.get(stage)
        if end_idx_result.get(stage) is not None else end_idx_map.get(stage)
        for stage in STAGE_ORDER
    }
    stage_max_depth = max(stage_depths.values()) if stage_depths else STAGE_SLOT_COUNT
    if stage_max_depth <= 0:
        stage_max_depth = STAGE_SLOT_COUNT

    patched_rel = os.path.relpath(result["patched_path"], OUTPUT_DIR)
    baseline_path_res = result.get("baseline_path") if result else None
    baseline_url = None
    if baseline_path_res:
        try:
            rel_baseline = os.path.relpath(baseline_path_res, OUTPUT_DIR)
            baseline_url = url_for("serve_image", filename=rel_baseline)
        except Exception:
            baseline_url = None

    curve_pts = []
    curve_path = ""
    try:
        baseline = torch.nn.SiLU()
        new_act = psy.make_base_act(opts["act"])
        sampler = psy.PsyAct(baseline, new_act, tau=opts["tau"], beta=opts["beta"], gamma=opts["gamma"])
        xs = torch.linspace(-6.0, 6.0, steps=121)
        ys = sampler(xs).detach().cpu()
        path_segments = []
        for xi, yi in zip(xs.tolist(), ys.tolist()):
            curve_pts.append({"x": xi, "y": yi})
            _x = (xi + 6.0) / 12.0 * 320.0
            _y = 260.0 - ((yi + 6.0) / 12.0 * 260.0)
            cmd = "M" if not path_segments else "L"
            path_segments.append(f"{cmd} {_x:.2f} {_y:.2f}")
        curve_path = " ".join(path_segments)
    except Exception:
        curve_pts = []
        curve_path = ""

    display = {
        "patched": url_for("serve_image", filename=patched_rel),
        "out_file": result["patched_path"],
        "patched_modules": result["patched_modules"],
        "used_fallback": result["used_fallback"],
        "reuse_baseline": result["reuse_baseline"],
        "stages": stages_str,
        "model_id": opts["model_id"],
        "model_label": model_info["label"],
        "curve": curve_pts,
        "curve_path": curve_path,
        "skip_baseline": opts["skip_baseline"],
        "stage_depths": stage_depths,
        "stage_max_depth": stage_max_depth,
        "start_idx_map": start_idx_display,
        "end_idx_map": end_idx_display,
        "step_window": result.get("step_window", {"start": opts["step_start"], "end": opts["step_end"]}),
        "baseline_url": baseline_url,
        "baseline_path": baseline_path_res,
    }
    return opts, display


# ---------------------- Routes ----------------------


@APP.route("/", methods=["GET", "POST"])
def index():
    message = None
    display = None
    opts = DEFAULT_OPTS.copy()
    if request.method == "POST":
        try:
            opts, display = _run_generation(request.form)
            if opts["skip_baseline"]:
                opts["calibrate"] = False
            start_map = display.get("start_idx_map", {}) if display else {}
            end_map = display.get("end_idx_map", {}) if display else {}
            step_window = display.get("step_window", {}) if display else {}
            def _fmt_range(stage: str) -> str:
                end_val = end_map.get(stage)
                end_str = "∞" if end_val is None else end_val
                return f"{start_map.get(stage, 0)}-{end_str}"
            step_start = step_window.get("start", 0)
            step_end = step_window.get("end")
            step_range = f"{step_start}-{step_end if step_end is not None else '∞'}"
            range_summary = f"↓{_fmt_range('down')} ∘{_fmt_range('mid')} ↑{_fmt_range('up')}"
            message = (f"{display['model_label']} — patched modules: {display['patched_modules']} "
                       f"(fallback={display['used_fallback']}, reuse_baseline={display['reuse_baseline']}, "
                       f"skip_baseline={display['skip_baseline']}, ranges={range_summary}, steps={step_range}).")
        except Exception as exc:
            message = f"Error: {exc}"

    return render_template_string(
        TEMPLATE,
        opts=opts,
        message=message,
        display=display,
        acts=["silu", "gelu", "gelu_tanh", "relu", "leakyrelu", "mish", "hswish", "softsign", "softplus", "hardtanh"],
        device=psy.device,
        model_choices=MODEL_CHOICES,
        stage_titles=STAGE_TITLES,
        stage_order=STAGE_ORDER,
        default_stage_depths=DEFAULT_STAGE_DEPTHS,
        default_max_depth=STAGE_SLOT_COUNT,
        default_start_idx_map={stage: DEFAULT_OPTS.get(f"start_idx_{stage}", 0) for stage in STAGE_ORDER},
        default_end_idx_map={stage: DEFAULT_OPTS.get(f"end_idx_{stage}") for stage in STAGE_ORDER},
    )


@APP.route("/images/<path:filename>")
def serve_image(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


# ---------------------- Template ----------------------

TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Psy UNet Lab</title>
    <style>
        body {
            font-family: "IBM Plex Sans", "Courier New", monospace;
            background: radial-gradient(circle at top left, #2b044d, #03020b 65%);
            color: #f0f0f0;
            margin: 0;
            padding: 0;
        }
        header {
            padding: 20px;
            text-align: center;
            background: linear-gradient(90deg, rgba(255,0,128,0.1), rgba(0,255,255,0.08), rgba(255,255,0,0.08));
            border-bottom: 1px solid rgba(255,255,255,0.12);
        }
        header h1 {
            margin: 0;
            font-size: 28px;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        main {
            display: flex;
            flex-wrap: wrap;
            padding: 24px;
            gap: 24px;
            justify-content: center;
        }
        .panel {
            background: rgba(12, 8, 24, 0.92);
            border: 2px solid rgba(255, 0, 204, 0.3);
            box-shadow: 0 0 18px rgba(255, 0, 204, 0.2);
            border-radius: 12px;
            padding: 28px;
            width: 460px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .panel form {
            width: 100%;
        }
        label {
            display: block;
            font-size: 13px;
            margin-bottom: 6px;
            text-transform: uppercase;
            color: #9df9ff;
        }
        input[type="text"],
        input[type="number"],
        textarea,
        select {
            width: 100%;
            padding: 8px;
            margin-bottom: 14px;
            background: rgba(25, 15, 40, 0.85);
            border: 1px solid rgba(0, 255, 255, 0.25);
            color: #fdfdfd;
            border-radius: 3px;
        }
        textarea {
            min-height: 60px;
            resize: vertical;
        }
        .stage-controls {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 14px;
        }
        .stage-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .stage-row label {
            display: flex;
            align-items: center;
            gap: 6px;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.8px;
            color: #9df9ff;
            min-width: 90px;
        }
        .stage-row.disabled {
            opacity: 0.6;
        }
        .stage-row.disabled input[type="number"] {
            opacity: 0.35;
        }
        .stage-inputs {
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }
        .stage-input {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .stage-input span {
            font-size: 10px;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            color: rgba(157, 249, 255, 0.7);
        }
        .stage-input input[type="number"] {
            width: 80px;
            margin: 0;
        }
        .stage-input input[type="number"][disabled] {
            opacity: 0.45;
            cursor: not-allowed;
        }
        .stage-row.disabled {
            opacity: 0.6;
        }
        .stage-row.disabled input[type="number"] {
            opacity: 0.35;
        }
        .checkbox-group {
            display: flex;
            gap: 12px;
            margin-bottom: 14px;
        }
        .checkbox-group label {
            display: flex;
            align-items: center;
            gap: 6px;
            text-transform: none;
            color: #f3f3f3;
        }
        .submit-btn {
            width: 100%;
            padding: 10px 16px;
            background: linear-gradient(90deg, #ff00cc, #00ffee);
            border: none;
            color: #03010a;
            font-weight: bold;
            letter-spacing: 1px;
            cursor: pointer;
            border-radius: 4px;
            text-transform: uppercase;
        }
        .submit-btn:hover {
            filter: brightness(1.1);
        }
        .message {
            margin-top: 12px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-left: 3px solid rgba(0, 255, 255, 0.6);
        }
        .results {
            flex: 1 1 520px;
            max-width: 720px;
            background: rgba(10, 6, 24, 0.92);
            border-radius: 12px;
            border: 2px solid rgba(0, 255, 204, 0.25);
            padding: 24px;
            box-shadow: 0 0 18px rgba(0, 255, 255, 0.25);
            position: relative;
            overflow: hidden;
        }
        .results h2 {
            margin-top: 0;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 18px;
        }
        .preview-single {
            margin: 0;
            text-align: center;
        }
        .preview-single img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .meta {
            font-size: 12px;
            margin-top: 16px;
            color: #f6f6f6;
        }
        .network-visual {
            margin-top: 26px;
            padding: 16px 18px 12px;
            border-radius: 10px;
            border: 1px solid rgba(0, 255, 255, 0.15);
            background: rgba(8, 4, 24, 0.72);
        }
        .network-visual h3 {
            margin: 0 0 12px;
            font-size: 14px;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #9df9ff;
        }
        .stage-axis {
            display: grid;
            gap: 6px;
            align-items: end;
            margin-bottom: 12px;
        }
        .stage-axis .axis-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: rgba(157, 249, 255, 0.7);
            grid-column: 1 / 2;
        }
        .stage-axis .axis-tick {
            font-size: 10px;
            text-align: center;
            color: rgba(255, 255, 255, 0.55);
        }
        .stage-axis .axis-caption {
            grid-column: 2 / -1;
            justify-self: center;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: rgba(157, 249, 255, 0.55);
            margin-top: 4px;
        }
        .network-stage {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            opacity: 0.55;
            transition: opacity 0.2s ease;
        }
        .network-stage:last-of-type {
            margin-bottom: 0;
        }
        .network-stage.selected {
            opacity: 1;
        }
        .stage-label {
            width: 56px;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.8px;
            color: rgba(157, 249, 255, 0.8);
        }
        .stage-track {
            flex: 1;
            display: grid;
            gap: 6px;
        }
        .stage-node {
            height: 12px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.08);
            position: relative;
            overflow: hidden;
        }
        .stage-node .node-index {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            letter-spacing: 0.5px;
            color: rgba(255, 255, 255, 0.45);
            pointer-events: none;
        }
        .stage-node.active .node-index {
            color: rgba(5, 8, 12, 0.75);
            font-weight: 600;
        }
        .stage-node::after {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: inherit;
            opacity: 0;
            transition: opacity 0.2s ease;
            background: linear-gradient(90deg, rgba(255,0,204,0.85), rgba(0,255,238,0.85));
        }
        .stage-node.active::after {
            opacity: 1;
        }
        .stage-node.stage-node-empty::after {
            opacity: 0;
        }
        .stage-node.stage-node-empty .node-index {
            color: rgba(255, 255, 255, 0.25);
        }
        .network-meta {
            margin-top: 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px 18px;
            font-size: 11px;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            color: rgba(246, 246, 246, 0.7);
        }
        .network-meta strong {
            color: #fdfdfd;
            margin-left: 4px;
        }
        .network-note {
            margin: 10px 0 0;
            font-size: 10px;
            color: rgba(200, 216, 255, 0.55);
            letter-spacing: 0.5px;
        }
        .indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 2px 8px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.06);
        }
        .indicator.active {
            background: rgba(0, 255, 255, 0.12);
            color: #00ffee;
        }
        .submit-btn:disabled {
            opacity: 0.7;
            cursor: wait;
        }
        .loading-status {
            display: none;
            margin-top: 16px;
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid rgba(0, 255, 255, 0.2);
            background: rgba(8, 4, 24, 0.75);
            color: #a6fdf9;
            font-size: 12px;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            align-items: center;
            gap: 10px;
        }
        .loading-status.active {
            display: flex;
        }
        .spinner {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            border: 3px solid rgba(0, 255, 255, 0.2);
            border-top-color: #ff00cc;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header>
        <h1>Psy UNet Control Pad</h1>
        <p>Device: {{ device.upper() }} &nbsp;|&nbsp; Quick knob sandbox for sd_unet_psy_acts</p>
    </header>
    <main>
        <section class="panel">
            <form method="POST" id="psy-form">
                <label for="prompt">Prompt</label>
                <textarea name="prompt" id="prompt">{{ opts.prompt }}</textarea>

                <label for="negative_prompt">Negative Prompt</label>
                <textarea name="negative_prompt" id="negative_prompt">{{ opts.negative_prompt }}</textarea>

                <label for="act">Activation</label>
                <select name="act" id="act">
                    {% for act in acts %}
                        <option value="{{ act }}" {% if opts.act == act %}selected{% endif %}>{{ act }}</option>
                    {% endfor %}
                </select>

                <label>Stages</label>
                <div class="stage-controls">
                    {% for stage in stage_order %}
                    {% set start_val = opts['start_idx_' + stage] %}
                    {% set end_val = opts['end_idx_' + stage] %}
                    <div class="stage-row" data-stage-control="{{ stage }}">
                        <label class="stage-option">
                            <input type="checkbox" name="stages" value="{{ stage }}" {% if stage in opts.stages %}checked{% endif %}>
                            {{ stage_titles.get(stage, stage) }}
                        </label>
                        <div class="stage-inputs">
                            <div class="stage-input">
                                <span>Start</span>
                                <input type="hidden" name="start_idx_{{ stage }}_hidden" value="{{ start_val }}">
                                <input type="number" min="0" name="start_idx_{{ stage }}" value="{{ start_val }}" {% if stage not in opts.stages %}disabled{% endif %}>
                            </div>
                            <div class="stage-input">
                                <span>End</span>
                                <input type="hidden" name="end_idx_{{ stage }}_hidden" value="{{ '' if end_val is none else end_val }}">
                                <input type="number" min="0" name="end_idx_{{ stage }}" value="{{ '' if end_val is none else end_val }}" placeholder="∞" {% if stage not in opts.stages %}disabled{% endif %}>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>

                <label for="tau">τ (curvature)</label>
                <input type="number" step="0.01" name="tau" id="tau" value="{{ opts.tau }}">

                <label for="beta">β (identity blend)</label>
                <input type="number" step="0.01" name="beta" id="beta" value="{{ opts.beta }}">

                <label for="gamma">γ (SiLU blend)</label>
                <input type="number" step="0.01" name="gamma" id="gamma" value="{{ opts.gamma }}">

                <label for="steps">Denoising Steps</label>
                <input type="number" name="steps" id="steps" value="{{ opts.steps }}">

                <div class="stage-inputs" style="margin-bottom:14px;">
                    <div class="stage-input">
                        <span>Step Start</span>
                        <input type="number" min="0" name="step_start" id="step_start" value="{{ opts.step_start }}">
                    </div>
                    <div class="stage-input">
                        <span>Step End</span>
                        <input type="number" min="0" name="step_end" id="step_end" value="{{ '' if opts.step_end is none else opts.step_end }}" placeholder="∞">
                    </div>
                </div>

                <label for="cfg">CFG Scale</label>
                <input type="number" step="0.1" name="cfg" id="cfg" value="{{ opts.cfg }}">

                <label for="seed">Seed</label>
                <input type="number" name="seed" id="seed" value="{{ opts.seed }}">

                <label for="height">Height (px)</label>
                <input type="number" name="height" id="height" value="{{ opts.height }}">

                <label for="width">Width (px)</label>
                <input type="number" name="width" id="width" value="{{ opts.width }}">

                <label for="model_id">Model</label>
                <select name="model_id" id="model_id">
                    {% for model in model_choices %}
                        <option value="{{ model.id }}" {% if opts.model_id == model.id %}selected{% endif %}>{{ model.label }}</option>
                    {% endfor %}
                </select>

                <div class="checkbox-group">
                    <label><input type="checkbox" name="patch_mlp" {% if opts.patch_mlp %}checked{% endif %}>Patch attention MLP</label>
                    <label><input type="checkbox" name="calibrate" {% if opts.calibrate %}checked{% endif %}>Variance match</label>
                    <label><input type="checkbox" name="skip_baseline" {% if opts.skip_baseline %}checked{% endif %}>Skip baseline run</label>
                </div>

                <button class="submit-btn" type="submit">Render</button>

                <div id="loading-indicator" class="loading-status" aria-live="polite" aria-busy="false">
                    <div class="spinner"></div>
                    <span>Rendering psychedelic layers...</span>
                </div>

                {% if message %}
                    <div class="message">{{ message }}</div>
                {% endif %}
            </form>
        </section>

        <section class="results" id="preview-panel">
            <h2>Preview</h2>
            <figure class="preview-single" id="curve-figure" style="margin-bottom:24px;">
                <figcaption>Activation Curve</figcaption>
                <svg viewBox="0 0 400 300" preserveAspectRatio="xMidYMid meet" style="width:100%;max-width:480px;border:1px solid rgba(255,255,255,0.15);background:rgba(0,0,0,0.15);">
                    <defs>
                        <linearGradient id="curveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" style="stop-color:#ff00cc;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#00ffee;stop-opacity:1" />
                        </linearGradient>
                    </defs>
                    <g transform="translate(40,20)">
                        {% for rx in range(0, 7) %}
                            <line x1="{{ rx*53.33 }}" y1="0" x2="{{ rx*53.33 }}" y2="260" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
                        {% endfor %}
                        {% for ry in range(0, 6) %}
                            <line x1="0" y1="{{ ry*52.0 }}" x2="320" y2="{{ ry*52.0 }}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
                        {% endfor %}
                        <line x1="0" y1="130" x2="320" y2="130" stroke="rgba(255,255,255,0.25)" stroke-width="1"/>
                        <line x1="160" y1="0" x2="160" y2="260" stroke="rgba(255,255,255,0.25)" stroke-width="1"/>
                        <path id="curve-path" d="{{ display.curve_path if display else '' }}" fill="none" stroke="url(#curveGradient)" stroke-width="2.5" />
                    </g>
                </svg>
            </figure>
            <div class="network-visual" id="network-visual">
                <h3>Patch Coverage (schematic)</h3>
                {% set stage_depths_map = display.stage_depths if display else default_stage_depths %}
                {% set start_idx_map = display.start_idx_map if display else default_start_idx_map %}
                {% set end_idx_display = display.end_idx_map if display else default_end_idx_map %}
                {% set down_end = end_idx_display.get('down') %}
                {% set mid_end = end_idx_display.get('mid') %}
                {% set up_end = end_idx_display.get('up') %}
                {% set down_end_label = down_end if down_end is not none else '∞' %}
                {% set mid_end_label = mid_end if mid_end is not none else '∞' %}
                {% set up_end_label = up_end if up_end is not none else '∞' %}
                {% set max_depth = display.stage_max_depth if display else default_max_depth %}
                {% set max_depth = max_depth if max_depth > 0 else 1 %}
                <div class="stage-axis" style="grid-template-columns: 56px repeat({{ max_depth }}, 1fr);">
                    <span class="axis-label">Stage</span>
                    {% for slot in range(max_depth) %}
                        <span class="axis-tick">{{ slot }}</span>
                    {% endfor %}
                    <span class="axis-caption">Residual block index (0 = shallow)</span>
                </div>
                {% for stage in stage_order %}
                {% set depth = stage_depths_map.get(stage, 0) %}
                {% set node_count = depth if depth > 0 else 1 %}
                <div class="network-stage" data-stage="{{ stage }}">
                    <span class="stage-label">{{ stage_titles.get(stage, stage) }}</span>
                    <div class="stage-track" style="grid-template-columns: repeat({{ node_count }}, 1fr);">
                        {% if depth > 0 %}
                            {% for i in range(depth) %}
                            <span class="stage-node" data-index="{{ i }}"><span class="node-index">{{ i }}</span></span>
                            {% endfor %}
                        {% else %}
                            <span class="stage-node stage-node-empty" data-index="0"><span class="node-index">—</span></span>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
                <div class="network-meta">
                    <span>Ranges<strong id="stage-range-display">↓ {{ start_idx_map.get('down', 0) }}–{{ down_end_label }} · ∘ {{ start_idx_map.get('mid', 0) }}–{{ mid_end_label }} · ↑ {{ start_idx_map.get('up', 0) }}–{{ up_end_label }}</strong></span>
                    <span class="indicator {% if opts.patch_mlp %}active{% endif %}" id="patch-mlp-display">
                        MLP patch<strong>{{ "ON" if opts.patch_mlp else "OFF" }}</strong>
                    </span>
                    <span>Depths<strong>↓ {{ stage_depths_map.get('down', 0) }} · ∘ {{ stage_depths_map.get('mid', 0) }} · ↑ {{ stage_depths_map.get('up', 0) }}</strong></span>
                </div>
                <p class="network-note">ResNet block counts pull from the active UNet (↓ {{ stage_depths_map.get('down', 0) }}, ∘ {{ stage_depths_map.get('mid', 0) }}, ↑ {{ stage_depths_map.get('up', 0) }}); highlights show patches applied between the stage’s start and end bounds.</p>
            </div>
            {% if display %}
                <figure class="preview-single">
                    <figcaption>Patched Output</figcaption>
                    <img src="{{ display.patched }}" alt="Patched image">
                </figure>
                {% if display.baseline_url %}
                <figure class="preview-single" style="margin-top:24px;">
                    <figcaption>Baseline Output {% if display.reuse_baseline %}(cached){% endif %}</figcaption>
                    <img src="{{ display.baseline_url }}" alt="Baseline image">
                </figure>
                {% endif %}
                <div class="meta">
                    <p>Model: {{ display.model_label }} <span style="opacity:0.7;">({{ display.model_id }})</span></p>
                    <p>Stages: {{ display.stages }} | Patched modules: {{ display.patched_modules }} | Fallback: {{ display.used_fallback }} | Reused baseline: {{ display.reuse_baseline }} | Skip baseline: {{ display.skip_baseline }}</p>
                    <p>Output file: <code>{{ display.out_file }}</code></p>
                    {% if display.baseline_path %}
                        <p>Baseline file: <code>{{ display.baseline_path }}</code></p>
                    {% endif %}
                    {% set step_window = display.step_window if display.step_window else {'start': opts.step_start, 'end': opts.step_end} %}
                    <p>Step window: <code>{{ step_window.start }} – {{ step_window.end if step_window.end is not none else '∞' }}</code></p>
                </div>
            {% else %}
                <p style="opacity:0.6;">No render yet &mdash; set knobs and click Render.</p>
            {% endif %}
        </section>
    </main>
    <script>
        const form = document.getElementById("psy-form");
        const storageKey = "psy-unet-control-pad";
        const initialCurve = {{ (display.curve if display else None) | tojson | safe }};
        const curvePathEl = document.getElementById("curve-path");
        const CURVE_FIELDS = new Set(["act", "tau", "beta", "gamma"]);
        const PATCH_FIELDS = new Set(["stages", "patch_mlp"]);
        const submitBtn = form ? form.querySelector(".submit-btn") : null;
        const loadingIndicator = document.getElementById("loading-indicator");
        const patchMlpDisplay = document.getElementById("patch-mlp-display");
        const STAGE_ORDER = {{ stage_order | tojson | safe }};
        const initialStageStarts = {{ (display.start_idx_map if display else default_start_idx_map) | tojson | safe }};
        const initialStageEnds = {{ (display.end_idx_map if display else default_end_idx_map) | tojson | safe }};
        const stageElements = {};
        const stageRows = {};
        const stageStartInputs = {};
        const stageHiddenInputs = {};
        const stageEndInputs = {};
        const stageEndHiddenInputs = {};
        const STAGE_START_FIELDS = new Set(STAGE_ORDER.map((stage) => `start_idx_${stage}`));
        const STAGE_END_FIELDS = new Set(STAGE_ORDER.map((stage) => `end_idx_${stage}`));
        const INFINITY_SYMBOL = "∞";
        const stageRangeDisplay = document.getElementById("stage-range-display");
        STAGE_ORDER.forEach((stageName) => {
            const el = document.querySelector(`.network-stage[data-stage="${stageName}"]`);
            if (el) stageElements[stageName] = el;
            const row = document.querySelector(`[data-stage-control="${stageName}"]`);
            if (row) stageRows[stageName] = row;
            stageStartInputs[stageName] = form ? form.elements[`start_idx_${stageName}`] : null;
            stageHiddenInputs[stageName] = form ? form.elements[`start_idx_${stageName}_hidden`] : null;
            stageEndInputs[stageName] = form ? form.elements[`end_idx_${stageName}`] : null;
            stageEndHiddenInputs[stageName] = form ? form.elements[`end_idx_${stageName}_hidden`] : null;
        });

        function formToJson() {
            const data = {};
            const formData = new FormData(form);
            for (const [key, value] of formData.entries()) {
                if (data[key]) {
                    if (!Array.isArray(data[key])) data[key] = [data[key]];
                    data[key].push(value);
                } else {
                    data[key] = value;
                }
            }
            data["step_start"] = formData.get("step_start") ?? "0";
            const rawStepEnd = formData.get("step_end");
            data["step_end"] = rawStepEnd === "" ? null : rawStepEnd;
            data["patch_mlp"] = formData.has("patch_mlp");
            data["calibrate"] = formData.has("calibrate");
            data["skip_baseline"] = formData.has("skip_baseline");
            for (const stage of STAGE_ORDER) {
                const hidden = stageHiddenInputs[stage];
                if (hidden) {
                    data[`start_idx_${stage}`] = hidden.value;
                }
                const endHidden = stageEndHiddenInputs[stage];
                if (endHidden) {
                    data[`end_idx_${stage}`] = endHidden.value;
                }
            }
            return data;
        }

        function applyPrefs(data) {
            if (!data) return;
            for (const [key, value] of Object.entries(data)) {
                const el = form.elements[key];
                if (!el) continue;
                if (key === "step_end" && (value === null || value === undefined || value === "None")) {
                    el.value = "";
                    continue;
                }
                if (el.type === "checkbox") {
                    el.checked = !!value;
                } else if (el instanceof RadioNodeList || Array.isArray(value)) {
                    const values = Array.isArray(value) ? value : [value];
                    for (const v of values) {
                        const target = [...form.elements[key]].find((node) => node.value === v);
                        if (target) target.checked = true;
                    }
                } else {
                    el.value = value;
                }
            }
            for (const stage of STAGE_ORDER) {
                const key = `start_idx_${stage}`;
                if (!(key in data)) continue;
                const val = data[key];
                if (stageStartInputs[stage]) stageStartInputs[stage].value = val;
                if (stageHiddenInputs[stage]) stageHiddenInputs[stage].value = val;
            }
            for (const stage of STAGE_ORDER) {
                const key = `end_idx_${stage}`;
                if (!(key in data)) continue;
                const val = data[key];
                if (stageEndInputs[stage]) stageEndInputs[stage].value = val;
                if (stageEndHiddenInputs[stage]) stageEndHiddenInputs[stage].value = val;
            }
        }

        function parseOrFallback(value, fallback) {
            const num = parseFloat(value);
            return Number.isFinite(num) ? num : fallback;
        }

        function clampStartIndex(value) {
            const num = parseInt(value, 10);
            if (!Number.isFinite(num) || num < 0) return 0;
            return num;
        }

        function syncStageStart(stage, clampInput = false) {
            const input = stageStartInputs[stage];
            const hidden = stageHiddenInputs[stage];
            const raw = input && input.value !== "" ? input.value : hidden ? hidden.value : "0";
            const value = clampStartIndex(raw);
            if (hidden) hidden.value = String(value);
            if (clampInput && input && input.value !== String(value)) {
                input.value = String(value);
            }
            return value;
        }

        function syncStageEnd(stage, minValue, clampInput = false) {
            const input = stageEndInputs[stage];
            const hidden = stageEndHiddenInputs[stage];
            const rawInput = input && typeof input.value === "string" ? input.value.trim() : "";
            let raw = rawInput !== "" ? rawInput : null;
            if (raw === null && !clampInput && hidden && typeof hidden.value === "string" && hidden.value.trim() !== "") {
                raw = hidden.value.trim();
            }
            let value = null;
            if (raw !== null && raw !== undefined) {
                const parsed = parseInt(raw, 10);
                if (Number.isFinite(parsed)) {
                    value = Math.max(minValue, parsed);
                }
            }
            if (clampInput && hidden) {
                hidden.value = value === null ? "" : String(value);
            }
            if (clampInput && input) {
                const displayVal = value === null ? "" : String(value);
                if (input.value !== displayVal) {
                    input.value = displayVal;
                }
            }
            return value;
        }

        function clampTau(value) {
            if (!Number.isFinite(value) || value === 0) return 1.0;
            const absVal = Math.abs(value);
            if (absVal < 1e-4) {
                return value > 0 ? 1e-4 : -1e-4;
            }
            return value;
        }

        function erfApprox(x) {
            const sign = Math.sign(x);
            const ax = Math.abs(x);
            const t = 1 / (1 + 0.3275911 * ax);
            const y = 1 - (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
            return sign * y;
        }

        function silu(x) {
            const expNeg = Math.exp(-x);
            return x / (1 + expNeg);
        }

        function softplus(x) {
            if (x > 12) return x;
            if (x < -12) return Math.exp(x);
            return Math.log1p(Math.exp(x));
        }

        function mish(x) {
            return x * Math.tanh(softplus(x));
        }

        function computeActivation(kind, x) {
            switch (kind) {
                case "gelu":
                    return 0.5 * x * (1 + erfApprox(x / Math.sqrt(2)));
                case "gelu_tanh":
                    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
                case "relu":
                    return Math.max(0, x);
                case "leakyrelu":
                    return x >= 0 ? x : 0.05 * x;
                case "mish":
                    return mish(x);
                case "hswish": {
                    const gate = Math.min(1, Math.max(0, (x + 3) / 6));
                    return x * gate;
                }
                case "softsign":
                    return x / (1 + Math.abs(x));
                case "softplus":
                    return softplus(x);
                case "hardtanh":
                    return Math.max(-1, Math.min(1, x));
                case "silu":
                default:
                    return silu(x);
            }
        }

        function generateCurvePoints(kind, tau, beta, gamma) {
            const pts = [];
            const safeTau = clampTau(tau);
            const denom = (Math.abs(safeTau) < 1e-4) ? (safeTau >= 0 ? 1e-4 : -1e-4) : safeTau;
            const samples = 121;
            for (let i = 0; i < samples; i += 1) {
                const x = -6 + (12 * i) / (samples - 1);
                const scaled = x / denom;
                const base = silu(scaled);
                const target = computeActivation(kind, scaled);
                const mix = (1 - gamma) * base + gamma * target;
                const y = (1 - beta) * mix + beta * x;
                pts.push({ x, y });
            }
            return pts;
        }

        function pointsToPath(points) {
            if (!Array.isArray(points) || points.length === 0) return "";
            let d = "";
            for (let i = 0; i < points.length; i += 1) {
                const pt = points[i];
                const svgX = ((pt.x + 6) / 12) * 320;
                const svgY = 260 - ((pt.y + 6) / 12) * 260;
                d += `${i === 0 ? "M" : "L"} ${svgX.toFixed(2)} ${svgY.toFixed(2)} `;
            }
            return d.trim();
        }

        function setCurve(points) {
            if (!curvePathEl) return;
            curvePathEl.setAttribute("d", pointsToPath(points));
        }

        function updateCurve() {
            if (!form) return;
            const actField = form.elements["act"];
            const tauField = form.elements["tau"];
            const betaField = form.elements["beta"];
            const gammaField = form.elements["gamma"];
            if (!actField || !tauField || !betaField || !gammaField) return;
            const actKind = String(actField.value || "silu").toLowerCase();
            const tau = clampTau(parseOrFallback(tauField.value, 1.0));
            const beta = parseOrFallback(betaField.value, 0.0);
            const gamma = parseOrFallback(gammaField.value, 1.0);
            const points = generateCurvePoints(actKind, tau, beta, gamma);
            setCurve(points);
        }

        function getCheckedStages() {
            return new Set([...form.querySelectorAll('input[name="stages"]:checked')].map((el) => el.value));
        }

        function updatePatchViz() {
            if (!form) return;
            const selectedStages = getCheckedStages();
            const starts = {};
            const ends = {};
            const patchMlp = !!form.elements["patch_mlp"]?.checked;

            STAGE_ORDER.forEach((stage) => {
                const startVal = syncStageStart(stage, false);
                starts[stage] = startVal;
                const input = stageStartInputs[stage];
                const endInput = stageEndInputs[stage];
                const row = stageRows[stage];
                const selected = selectedStages.has(stage);
                if (input) {
                    input.disabled = !selected;
                    input.setAttribute("aria-disabled", String(!selected));
                }
                if (endInput) {
                    endInput.disabled = !selected;
                    endInput.setAttribute("aria-disabled", String(!selected));
                }
                if (row) {
                    row.classList.toggle("disabled", !selected);
                }
                const endVal = syncStageEnd(stage, startVal, false);
                ends[stage] = endVal;
                const stageEl = stageElements[stage];
                if (!stageEl) return;
                stageEl.classList.toggle("selected", selected);
                const nodes = stageEl.querySelectorAll(".stage-node");
                nodes.forEach((node) => {
                    const idx = Number(node.dataset.index || 0);
                    const withinEnd = ends[stage] === null || idx <= ends[stage];
                    const isActive = selected && idx >= startVal && withinEnd;
                    node.classList.toggle("active", isActive);
                });
            });

            if (stageRangeDisplay) {
                const formatRange = (stage) => {
                    const endVal = ends[stage];
                    const endLabel = (endVal === null || endVal === undefined) ? INFINITY_SYMBOL : endVal;
                    return `${starts[stage] ?? 0}–${endLabel}`;
                };
                const summary = `↓ ${formatRange('down')} · ∘ ${formatRange('mid')} · ↑ ${formatRange('up')}`;
                stageRangeDisplay.textContent = summary;
            }
            if (patchMlpDisplay) {
                patchMlpDisplay.classList.toggle("active", patchMlp);
                const strong = patchMlpDisplay.querySelector("strong");
                if (strong) {
                    strong.textContent = patchMlp ? "ON" : "OFF";
                }
            }
        }

        function handleFormChange(event) {
            if (!event || !event.target) return;
            const { name } = event.target;
            if (STAGE_START_FIELDS.has(name)) {
                const stage = name.replace("start_idx_", "");
                syncStageStart(stage, event.type === "change");
                updatePatchViz();
                return;
            }
            if (STAGE_END_FIELDS.has(name)) {
                const stage = name.replace("end_idx_", "");
                const startVal = syncStageStart(stage, false);
                syncStageEnd(stage, startVal, event.type === "change");
                updatePatchViz();
                return;
            }
            if (CURVE_FIELDS.has(name)) {
                updateCurve();
            }
            if (PATCH_FIELDS.has(name)) {
                updatePatchViz();
            }
        }

        function showLoading() {
            if (loadingIndicator) {
                loadingIndicator.classList.add("active");
                loadingIndicator.setAttribute("aria-busy", "true");
            }
            if (submitBtn) {
                submitBtn.disabled = true;
                if (!submitBtn.dataset.originalLabel) {
                    submitBtn.dataset.originalLabel = submitBtn.textContent ?? "Render";
                }
                submitBtn.textContent = "Rendering…";
            }
        }

        document.addEventListener("DOMContentLoaded", () => {
            try {
                const stored = localStorage.getItem(storageKey);
                if (stored) {
                    applyPrefs(JSON.parse(stored));
                }
            } catch (err) {
                console.warn("Unable to parse stored prefs", err);
            }
            STAGE_ORDER.forEach((stage) => {
                const initial = clampStartIndex(initialStageStarts[stage] ?? 0);
                if (stageHiddenInputs[stage] && stageHiddenInputs[stage].value === "") {
                    stageHiddenInputs[stage].value = String(initial);
                }
                if (stageStartInputs[stage] && stageStartInputs[stage].value === "") {
                    stageStartInputs[stage].value = String(initial);
                }
                const initialEnd = initialStageEnds[stage];
                const endHidden = stageEndHiddenInputs[stage];
                const endInput = stageEndInputs[stage];
                const endDisplay = initialEnd === null || initialEnd === undefined ? "" : String(Math.max(initial, initialEnd));
                if (endHidden && endHidden.value === "") {
                    endHidden.value = endDisplay;
                }
                if (endInput && endInput.value === "") {
                    endInput.value = endDisplay;
                }
            });
            STAGE_ORDER.forEach((stage) => {
                syncStageStart(stage, false);
            });
            STAGE_ORDER.forEach((stage) => {
                const startVal = syncStageStart(stage, false);
                syncStageEnd(stage, startVal, false);
            });
            if (loadingIndicator) {
                loadingIndicator.classList.remove("active");
                loadingIndicator.setAttribute("aria-busy", "false");
            }
            if (submitBtn) {
                submitBtn.disabled = false;
                if (submitBtn.dataset.originalLabel) {
                    submitBtn.textContent = submitBtn.dataset.originalLabel;
                }
            }
            if (Array.isArray(initialCurve) && initialCurve.length) {
                setCurve(initialCurve);
            }
            updateCurve();
            updatePatchViz();
        });

        form.addEventListener("input", handleFormChange);
        form.addEventListener("change", handleFormChange);

        form.addEventListener("submit", () => {
            STAGE_ORDER.forEach((stage) => {
                const startVal = syncStageStart(stage, true);
                syncStageEnd(stage, startVal, true);
            });
            showLoading();
            try {
                localStorage.setItem(storageKey, JSON.stringify(formToJson()));
            } catch (err) {
                console.warn("Unable to persist prefs", err);
            }
        });
    </script>
</body>
</html>
"""


# ---------------------- Entrypoint ----------------------

if __name__ == "__main__":
    print("Launching Psy UNet Control Pad at http://127.0.0.1:7860")
    APP.run(host="127.0.0.1", port=7860, debug=False)
