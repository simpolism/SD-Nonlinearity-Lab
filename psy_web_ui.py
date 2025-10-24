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
from typing import Any, Dict, Tuple

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

# ---------------------- Helpers ----------------------


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
    return PIPELINE_CACHE[key]


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
        tuple(sorted(opts["stages"])),
        opts["start_idx"],
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
        "start_idx": int(form.get("start_idx", DEFAULT_OPTS["start_idx"])),
        "patch_mlp": form.get("patch_mlp") == "on",
        "calibrate": form.get("calibrate") == "on",
        "height": int(form.get("height", DEFAULT_OPTS["height"])) if form.get("height") else None,
        "width": int(form.get("width", DEFAULT_OPTS["width"])) if form.get("width") else None,
        "skip_baseline": form.get("skip_baseline") == "on",
    }

    sel_model_id = form.get("model_id", DEFAULT_OPTS["model_id"])
    model_info = next((m for m in MODEL_CHOICES if m["id"] == sel_model_id), MODEL_CHOICES[0])
    opts["model_id"] = model_info["id"]
    opts["pipeline"] = model_info["pipeline"]
    opts["dtype"] = model_info["dtype"]

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
        patch_mlp=opts["patch_mlp"],
        calibrate=opts["calibrate"],
        height=opts["height"],
        width=opts["width"],
        model_id=opts["model_id"],
        pipeline=opts["pipeline"],
        dtype=opts["dtype"],
        skip_baseline=opts["skip_baseline"],
    )

    patched_rel = os.path.relpath(result["patched_path"], OUTPUT_DIR)

    curve_pts = []
    try:
        baseline = torch.nn.SiLU()
        new_act = psy.make_base_act(opts["act"])
        sampler = psy.PsyAct(baseline, new_act, tau=opts["tau"], beta=opts["beta"], gamma=opts["gamma"])
        xs = torch.linspace(-6.0, 6.0, steps=121)
        ys = sampler(xs).detach().cpu()
        for xi, yi in zip(xs.tolist(), ys.tolist()):
            curve_pts.append({"x": xi, "y": yi})
    except Exception:
        curve_pts = []

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
            message = (f"{display['model_label']} — patched modules: {display['patched_modules']} "
                       f"(fallback={display['used_fallback']}, reuse_baseline={display['reuse_baseline']}, "
                       f"skip_baseline={display['skip_baseline']}).")
        except Exception as exc:
            message = f"Error: {exc}"

    return render_template_string(
        TEMPLATE,
        opts=opts,
        message=message,
        display=display,
        acts=["silu", "gelu", "gelu_tanh", "relu", "leakyrelu", "mish", "hswish", "softsign", "softplus"],
        device=psy.device,
        model_choices=MODEL_CHOICES,
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
                <div class="checkbox-group">
                    {% for stage in ['down','mid','up'] %}
                        <label><input type="checkbox" name="stages" value="{{ stage }}" {% if stage in opts.stages %}checked{% endif %}>{{ stage }}</label>
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
                <label for="start_idx">Stage Start Index</label>
                <input type="number" name="start_idx" id="start_idx" value="{{ opts.start_idx }}">

                <div class="checkbox-group">
                    <label><input type="checkbox" name="patch_mlp" {% if opts.patch_mlp %}checked{% endif %}>Patch attention MLP</label>
                    <label><input type="checkbox" name="calibrate" {% if opts.calibrate %}checked{% endif %}>Variance match</label>
                    <label><input type="checkbox" name="skip_baseline" {% if opts.skip_baseline %}checked{% endif %}>Skip baseline run</label>
                </div>

                <button class="submit-btn" type="submit">Render</button>

                {% if message %}
                    <div class="message">{{ message }}</div>
                {% endif %}
            </form>
        </section>

        <section class="results" id="preview-panel">
            <h2>Preview</h2>
            {% if display %}
                <figure class="preview-single">
                    <figcaption>Patched Output</figcaption>
                    <img src="{{ display.patched }}" alt="Patched image">
                </figure>
                <div class="meta">
                    <p>Model: {{ display.model_label }} <span style="opacity:0.7;">({{ display.model_id }})</span></p>
                    <p>Stages: {{ display.stages }} | Patched modules: {{ display.patched_modules }} | Fallback: {{ display.used_fallback }} | Reused baseline: {{ display.reuse_baseline }} | Skip baseline: {{ display.skip_baseline }}</p>
                    <p>Output file: <code>{{ display.out_file }}</code></p>
                </div>
                {% if display.curve %}
                <figure class="preview-single" style="margin-top:24px;">
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
                            <path d="{% for pt in display.curve %}{% set _x = (pt.x + 6) / 12.0 * 320 %}{% set _y = 260 - ((pt.y + 6) / 12.0 * 260) %}{% if loop.first %}M {{ '%.2f' % _x }} {{ '%.2f' % _y }}{% else %} L {{ '%.2f' % _x }} {{ '%.2f' % _y }}{% endif %}{% endfor %}" fill="none" stroke="url(#curveGradient)" stroke-width="2.5" />
                        </g>
                    </svg>
                </figure>
                {% endif %}
            {% else %}
                <p style="opacity:0.6;">No render yet &mdash; set knobs and click Render.</p>
            {% endif %}
        </section>
    </main>
    <script>
        const form = document.getElementById("psy-form");
        const storageKey = "psy-unet-control-pad";

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
            data["patch_mlp"] = formData.has("patch_mlp");
            data["calibrate"] = formData.has("calibrate");
            data["skip_baseline"] = formData.has("skip_baseline");
            return data;
        }

        function applyPrefs(data) {
            if (!data) return;
            for (const [key, value] of Object.entries(data)) {
                const el = form.elements[key];
                if (!el) continue;
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
        });

        form.addEventListener("submit", () => {
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
