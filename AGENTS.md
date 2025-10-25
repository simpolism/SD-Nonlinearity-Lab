# SD Nonlinearity Lab – Agent Guide

This document is aimed at any automated assistant or scripted workflow that operates inside this repository.  Use it to understand the preferred entry points, constraints, and coordination rules.

---

## 1. Mission Snapshot

SD Nonlinearity Lab is an experimentation harness for exploring different nonlinearities in Stable Diffusion UNet activations at inference time.  Two surfaces matter:

1. `sd_unet_psy_acts.py` – CLI/SDK for patching activations, caching baselines, and variance-matching.
2. `psy_web_ui.py` – Flask web application that wraps the CLI in a controls dashboard (per-stage start/end, step windows, baseline previews).

Agents typically:

- Generate or compare A/B renders for a fixed seed.
- Sweep activation parameters (`act`, `tau`, `beta`, `gamma`).
- Toggle stage coverage (`down`, `mid`, `up`) and depth windows.
- Gate patched activations to a denoising step window.
- Launch or interact with the web UI locally (default port `7860`).

---

## 2. Safety & Resource Rules

| Topic | Expectations |
|-------|--------------|
| GPU usage | Pipelines load in FP16 by default. Idle pipelines are offloaded back to CPU to limit VRAM. |
| Baselines | Baseline renders are reused whenever prompt/steps/cfg/seed/model are unchanged. Stage/step tweaks do **not** force a recompute. |
| Caching | Baseline cache lives under `web_out/baseline_cache/`. Only remove entries intentionally – new baselines auto-replace older ones when the key collides. |
| External writes | Avoid writing outside the repo except for HuggingFace model downloads (handled by diffusers). |
| Secrets | No API keys are stored here. If an agent requires authentication (e.g., HF token) it must obtain it from environment variables supplied at runtime. |

---

## 3. Launch Recipes

### 3.1 CLI Rendering

```bash
python sd_unet_psy_acts.py \
  --prompt "dreamlike cityscape" \
  --out city.png \
  --act mish --tau 1.4 --beta 0.25 --gamma 0.8 \
  --stages up,mid \
  --start-idx-up 2 --end-idx-up 9 \
  --step-start 10 --step-end 25 \
  --steps 30 --cfg 7.5 --seed 1337 \
  --calibrate
```

Notes for agents:
- `--step-start/-end` gate when patched activations are enabled.
- Per-stage overrides fall back to `--start-idx` / `--end-idx` when omitted.
- If calibration is requested and a cached baseline exists, it will be reused; otherwise one is generated automatically.

### 3.2 Web UI

```bash
python psy_web_ui.py
# opens http://127.0.0.1:7860
```

- The UI displays real-time activation curves and stage coverage.
- Step and stage ranges accept “∞” by leaving the field blank.
- Baseline previews appear beneath the patched render when available and indicate whether the baseline was reused from cache.

---

## 4. Coordination Guidelines

1. **Preserve determinism.** Always specify `--seed` (CLI) or set the seed field (UI) when comparing runs.  
2. **Respect cache semantics.** Only delete `web_out/baseline_cache/` or `web_out/` outputs if new baselines are explicitly required.  
3. **Record settings.** When sharing outputs, include activation knob values, stage/step ranges, and whether variance matching was enabled.  
4. **Testing.** Use `python -m py_compile psy_web_ui.py sd_unet_psy_acts.py` before submitting changes.  
5. **Notebooks/Sweeps.** `sweep_call_psy_generator.py` demonstrates programmatic sweeps—copy it when automating large parameter grids.  

---

## 5. Useful Paths & Environment

| Path | Purpose |
|------|---------|
| `sd_unet_psy_acts.py` | Core activation patching CLI. |
| `psy_web_ui.py` | Flask UI server. |
| `web_out/` | Generated images and cached baselines. |
| `saved_web_out/` | Optional archive of previous runs. |
| `sweep_call_psy_generator.py` | Legacy sweep driver for scripted parameter grids. |
| `requirements.txt` | Python dependencies; install via `pip install -r requirements.txt`. |

Environment tips:
- Requires Python ≥ 3.10 with CUDA-capable PyTorch for GPU acceleration.
- HuggingFace models download automatically the first time they are used.
- `torch.cuda.empty_cache()` is invoked when swapping pipelines; no manual action needed unless GPU pressure persists.

---

## 6. Extensibility Hooks

- Activation library: extend `make_base_act` for new nonlinearities.
- Step windowing: adjust `step_start` / `step_end` or modify the callback in `sd_unet_psy_acts.py` for custom schedules.
- Future img2img support (planned): pipeline swap will be keyed on optional init images; agents should pass file paths once implemented.
- Sweeps: `sweep_call_psy_generator.py` orchestrates older grid runs (global `--start-idx`, full step span). Extend it if you need per-stage ranges or step windows.

---

Stay within these guidelines and SD Nonlinearity Lab should remain stable, reproducible, and ready for activation explorations.
