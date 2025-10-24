# AGENTS.md

## Overview

This project explores **test-time manipulation of UNet nonlinearities** in a Stable-Diffusion-style model to simulate “psychedelic-like” network dynamics: we reduce the **curvature** of activation functions and/or blend in **identity passthrough**, optionally swapping the **activation type** itself. The core intuition is that flattening/gating less tightly allows atypical signals to propagate, increasing the entropy of internal states and producing images with altered coherence, texture, and semantics — an AI analogue to how psychedelics may modulate neuronal input-output curves.

## What the code does

- Loads `runwayml/stable-diffusion-v1-5` (fp16) and renders **A/B images** with the **same seed**:
  - **A**: baseline (unaltered activations).
  - **B**: patched UNet with your chosen activation **shape** and two “psychedelic” knobs.

- Three independent knobs:
  1) **Activation type** `--act`  
     Choose the target nonlinearity: `silu, gelu, gelu_tanh, relu, leakyrelu, mish, hswish, softsign, softplus`.

  2) **Curvature reduction** `--tau`  
     Applies the activation to a scaled preactivation `x/τ`. **τ>1** flattens curvature (weaker gating/saturation).

  3) **Identity blending** `--beta`  
     Blends the activation with identity: `y = (1-β)·act(x/τ) + β·x`. Higher **β** → more linear passthrough, looser constraints.

  Additionally, `--gamma` blends **SiLU** with your chosen activation **before** identity mixing:  
  `mix = (1-γ)·SiLU(x/τ) + γ·act_new(x/τ)`.

- **Targeting**: limit changes to `--stages` (`down, mid, up`) and `--start-idx` within those stages. `--patch-mlp` includes attention MLPs.

- **Variance matching** `--calibrate`: on a baseline pass, it records per-site mean/variance and then applies an affine correction to the patched activations so downstream layers see the same first/second moments. This isolates **shape** effects from **scale** artifacts.

- **Functional fallback**: some SD UNets call `F.silu`/`F.gelu` directly (no `nn.SiLU/nn.GELU` submodules). If no modules were replaced, the script automatically **monkey-patches** those functional calls during the patched pass only. This guarantees your knobs take effect across model variants.

## Why this design

- **Safe, reversible interventions**: no fine-tuning; A/B is strictly inference-time.
- **Comparability**: fixed seed and optional variance matching isolate the perceptual effects of changing **activation geometry**.
- **Scoping**: per-stage and per-depth selection lets you study how early (texture) versus late (semantics/layout) processing shifts under flatter/identity-blended activations.

## Quick start

```bash
python sd_unet_psy_acts.py \
  --prompt "a cozy reading nook with warm ambient light, 35mm film grain" \
  --out demo.png \
  --act mish --tau 1.3 --beta 0.2 --gamma 1.0 \
  --stages up,mid --steps 30 --cfg 7.5 --seed 12345 --calibrate
# -> demo_A_baseline.png and demo_B_patched.png
```

## Interpreting effects (rules of thumb)

* Increase **τ** (curvature flattening): freer flow of weak/atypical features → more surreal/global drift.
* Increase **β** (identity blend): stronger linear passthrough → higher entropy, looser constraints, potential loss of crispness.
* Change **act**: different tail/saturation behavior (e.g., `mish` richer tails; `gelu_tanh` smoother) reshapes style.

## Extending

* **Step schedules**: make τ/β a function of the denoising timestep (e.g., stronger early → “coming-up,” taper late).
* **Metrics**: pair with a sweep driver that computes LPIPS, CLIPScore, entropy, sharpness, and produces contact sheets.
* **Other backbones**: the same pattern applies to SD v2/XL UNets; just keep the functional fallback in place.

## Troubleshooting

* **Images look identical**: likely no modules were replaced; the script now auto-patches `F.silu/F.gelu`. If still identical, increase `--tau`/`--beta`, try `--act mish`, or target `--stages down`.
* **Washed-out results**: reduce `--beta`, enable `--calibrate`, or try a gentler τ (e.g., 1.1–1.2).
* **OOM on 12–16 GB**: keep batch=1, use `--steps 20–30`, and rely on attention slicing (enabled by default when available).
