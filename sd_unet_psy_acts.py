# sd_unet_psy_acts.py
# -*- coding: utf-8 -*-
"""
Drop-in UNet activation patcher for Stable Diffusion v1.5

Knobs:
  - --act {silu,gelu,gelu_tanh,relu,leakyrelu,mish,hswish,softsign,softplus}
  - --tau <float>        (pre-activation temperature; >1.0 flattens curvature)
  - --beta <float>       (identity blend; 0 = pure act, 1 = pure identity)
  - --gamma <float>      (blend new act with baseline SiLU before identity)
  - --stages down,mid,up (comma list)
  - --start-idx <int>    (first resblock index within targeted stages)
  - --patch-mlp          (also patch attention MLP GELU/SiLU)
  - --calibrate          (variance-match per patched module using a baseline pass)

Outputs:
  <OUT>_A_baseline.png and <OUT>_B_patched.png (same seed)
"""

import argparse, math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import ImageChops

torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- Activation Library ----------------------

def make_base_act(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "silu":        return nn.SiLU()
    if kind == "gelu":        return nn.GELU()  # erf approx
    if kind == "gelu_tanh":   return nn.GELU(approximate="tanh")
    if kind == "relu":        return nn.ReLU()
    if kind == "leakyrelu":   return nn.LeakyReLU(0.05)
    if kind == "mish":        return nn.Mish()
    if kind == "hswish":
        class HSwish(nn.Module):
            def forward(self, x): return x * (torch.clamp(x + 3, 0, 6) / 6)
        return HSwish()
    if kind == "softsign":    return nn.Softsign()
    if kind == "softplus":    return nn.Softplus(beta=1.0, threshold=20.0)
    raise ValueError(f"Unknown act kind: {kind}")

class PsyAct(nn.Module):
    """
    Psychedelic-style activation wrapper with 3 knobs + optional affine correction:
      mix = (1-gamma)*SiLU(x/tau) + gamma*new_act(x/tau)
      y   = (1-beta)*mix + beta*x
      out = gain*y + bias
    """
    def __init__(self, baseline: nn.Module, new_act: nn.Module,
                 tau: float = 1.0, beta: float = 0.0, gamma: float = 1.0,
                 gain: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.baseline = baseline
        self.new_act  = new_act
        self.tau      = float(tau)
        self.beta     = float(beta)
        self.gamma    = float(gamma)
        self.register_buffer("gain_buf", torch.tensor(float(gain)), persistent=False)
        self.register_buffer("bias_buf", torch.tensor(float(bias)), persistent=False)

    def forward(self, x):
        z = x / self.tau
        mix = (1.0 - self.gamma) * self.baseline(z) + self.gamma * self.new_act(z)
        y = (1.0 - self.beta) * mix + self.beta * x
        return self.gain_buf * y + self.bias_buf

# ---------------------- Functional Fallback ----------------------
# Covers models that call F.silu / F.gelu directly (no nn.SiLU/nn.GELU submodules).

class FuncPsyAct:
    """
    Functional version of the PsyAct logic for F.silu / F.gelu.
    No affine correction here (variance-match only supported for module path).
    """
    def __init__(self, baseline_fn: Callable, new_act_module: nn.Module,
                 tau=1.0, beta=0.0, gamma=1.0,
                 orig_silu: Callable = None, orig_gelu: Callable = None):
        self.baseline_fn = baseline_fn
        self.new_act  = new_act_module
        self.tau, self.beta, self.gamma = float(tau), float(beta), float(gamma)
        self.orig_silu = orig_silu
        self.orig_gelu = orig_gelu

    def __call__(self, x, *args, **kwargs):
        z = x / self.tau
        base = self.baseline_fn(z, *args, **kwargs)
        mix = (1.0 - self.gamma) * base + self.gamma * self._call_new(z)
        return (1.0 - self.beta) * mix + self.beta * x

    def _call_new(self, z):
        if self.orig_silu is None and self.orig_gelu is None:
            return self.new_act(z)
        cur_silu, cur_gelu = F.silu, F.gelu
        try:
            if self.orig_silu is not None:
                F.silu = self.orig_silu
            if self.orig_gelu is not None:
                F.gelu = self.orig_gelu
            return self.new_act(z)
        finally:
            F.silu, F.gelu = cur_silu, cur_gelu

class patch_functional_acts:
    """
    Context manager: temporarily patch F.silu and F.gelu with our functional PsyAct.
    """
    def __init__(self, act_kind: str, tau: float, beta: float, gamma: float):
        self.act_kind, self.tau, self.beta, self.gamma = act_kind, tau, beta, gamma
        self._old_silu = None
        self._old_gelu = None

    def __enter__(self):
        self._old_silu, self._old_gelu = F.silu, F.gelu
        new_act_silu = make_base_act(self.act_kind)
        new_act_gelu = make_base_act(self.act_kind)
        silu_fn = FuncPsyAct(
            self._old_silu, new_act_silu,
            tau=self.tau, beta=self.beta, gamma=self.gamma,
            orig_silu=self._old_silu, orig_gelu=self._old_gelu
        )
        gelu_fn = FuncPsyAct(
            self._old_gelu, new_act_gelu,
            tau=self.tau, beta=self.beta, gamma=self.gamma,
            orig_silu=self._old_silu, orig_gelu=self._old_gelu
        )
        F.silu = lambda x, *a, **k: silu_fn(x, *a, **k)
        F.gelu = lambda x, *a, **k: gelu_fn(x, *a, **k)  # keep attention MLPs coherent
        return self

    def __exit__(self, exc_type, exc, tb):
        F.silu, F.gelu = self._old_silu, self._old_gelu

# ---------------------- UNet Patching ----------------------

@dataclass
class PatchCfg:
    stages: List[str]
    start_idx: int
    act_kind: str
    tau: float
    beta: float
    gamma: float
    patch_attn_mlp: bool
    calibrate: bool

def stage_of_name(name: str):
    if ".down_blocks." in name: return "down"
    if ".up_blocks."   in name: return "up"
    if ".mid_block."   in name: return "mid"
    return None

def _iter_leaf_acts(module: nn.Module):
    # Yield (parent, attr, child, fullname) for leaves
    for fullname, m in module.named_modules():
        parent_name = fullname.rsplit('.', 1)[0] if '.' in fullname else ''
        parent = dict(module.named_modules()).get(parent_name, module if parent_name == '' else None)
        if parent is None:
            continue
        for attr, child in list(parent._modules.items()):
            if child is m and len(list(child.children())) == 0:
                yield parent, attr, child, fullname

def build_psy_factory(act_kind: str, tau: float, beta: float, gamma: float) -> Callable[[], nn.Module]:
    def factory():
        baseline = nn.SiLU()
        new_act  = make_base_act(act_kind)
        return PsyAct(baseline, new_act, tau=tau, beta=beta, gamma=gamma)
    return factory

def patch_unet(unet: nn.Module, cfg: PatchCfg) -> Tuple[int, Dict[str, PsyAct]]:
    from diffusers.models.resnet import ResnetBlock2D
    count = 0
    patched: Dict[str, PsyAct] = {}
    idx_by_stage = {"down": -1, "mid": -1, "up": -1}

    # Replace SiLU inside ResnetBlock2D (core UNet nonlinearity)
    for fullname, m in unet.named_modules():
        st = stage_of_name(fullname)
        if st and isinstance(m, ResnetBlock2D):
            idx_by_stage[st] += 1
            if st in cfg.stages and idx_by_stage[st] >= cfg.start_idx:
                for attr, child in list(m._modules.items()):
                    if isinstance(child, nn.SiLU):
                        psy = build_psy_factory(cfg.act_kind, cfg.tau, cfg.beta, cfg.gamma)()
                        m._modules[attr] = psy
                        patched[f"{fullname}.{attr}"] = psy
                        count += 1

    # Optionally patch attention MLP activations (often GELU/SiLU)
    if cfg.patch_attn_mlp:
        for parent, attr, child, fullname in _iter_leaf_acts(unet):
            st = stage_of_name(fullname)
            if st in (cfg.stages or []):
                if isinstance(child, (nn.GELU, nn.SiLU)):
                    psy = build_psy_factory(cfg.act_kind, cfg.tau, cfg.beta, cfg.gamma)()
                    parent._modules[attr] = psy
                    patched[fullname] = psy
                    count += 1

    return count, patched

# ---------------------- Variance Matching ----------------------

@torch.no_grad()
def variance_match(per_act: Dict[str, PsyAct],
                   records: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
    """
    records[name] = (pre_x, base_out) collected under baseline pass.
    After we have PsyAct in place (with same pre_x), compute new_out and set gain/bias
    so E[y] and Std[y] ≈ baseline's.
    """
    eps = 1e-6
    for name, psy in per_act.items():
        if name not in records:
            continue
        x, y_old = records[name]
        y_new = psy.forward(x)
        mu_old, sd_old = y_old.mean().item(), y_old.std().item()
        mu_new, sd_new = y_new.mean().item(), y_new.std().item()
        g = (sd_old / max(sd_new, eps))
        b = (mu_old - g * mu_new)
        psy.gain_buf.data.fill_(g)
        psy.bias_buf.data.fill_(b)

class ActTap(nn.Module):
    """Passthrough that records inputs/outputs during baseline pass."""
    def __init__(self, act: nn.Module, store: Dict[str, Tuple[torch.Tensor, torch.Tensor]], key: str):
        super().__init__()
        self.act = act
        self.store = store
        self.key = key
    def forward(self, x):
        y = self.act(x)
        if self.key not in self.store:
            self.store[self.key] = (x.detach(), y.detach())
        return y

def insert_taps_for_baseline(unet: nn.Module) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    store: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for parent, attr, child, fullname in _iter_leaf_acts(unet):
        if isinstance(child, (nn.SiLU, nn.GELU)):
            parent._modules[attr] = ActTap(child, store, fullname)
    return store

def remove_taps(unet: nn.Module):
    for parent, attr, child, fullname in _iter_leaf_acts(unet):
        if isinstance(child, ActTap):
            parent._modules[attr] = child.act

# ---------------------- Main Run ----------------------

def run(args):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    # ---- Baseline (and optional stat collection) ----
    g = torch.Generator(device=device).manual_seed(args.seed)
    baseline_store = {}
    if args.calibrate:
        baseline_store = insert_taps_for_baseline(pipe.unet)

    imgA = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=g
    ).images[0]
    imgA.save(args.out.replace(".png", "_A_baseline.png"))

    if args.calibrate:
        remove_taps(pipe.unet)

    # ---- Patch modules ----
    cfg = PatchCfg(
        stages=[s.strip() for s in args.stages.split(",") if s.strip()],
        start_idx=args.start_idx,
        act_kind=args.act,
        tau=args.tau,
        beta=args.beta,
        gamma=args.gamma,
        patch_attn_mlp=args.patch_mlp,
        calibrate=args.calibrate
    )
    n, per_act = patch_unet(pipe.unet, cfg)
    print(f"[UNet] Patched {n} activation modules | act={args.act} tau={args.tau} beta={args.beta} gamma={args.gamma} | stages={cfg.stages} start_idx={cfg.start_idx} MLP={cfg.patch_attn_mlp}")

    # ---- Variance-match (module path only) ----
    if args.calibrate and per_act:
        print("[Calib] Performing variance matching on patched modules...")
        variance_match(per_act, baseline_store)

    # ---- Patched image with the SAME seed ----
    g = torch.Generator(device=device).manual_seed(args.seed)
    if n == 0:
        print("[Fallback] No nn.SiLU/nn.GELU modules found; patching F.silu/F.gelu functionally.")
        with patch_functional_acts(args.act, args.tau, args.beta, args.gamma):
            imgB = pipe(
                args.prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=g
            ).images[0]
    else:
        imgB = pipe(
            args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=g
        ).images[0]

    outB = args.out.replace(".png", "_B_patched.png")
    imgB.save(outB)

    # ---- Sanity check: warn if unchanged ----
    if ImageChops.difference(imgA, imgB).getbbox() is None:
        print("[WARN] Patched image is pixel-identical to baseline. "
              "If you expected differences, increase --tau / --beta or verify patching was applied.")

    print("Done. Wrote:", args.out.replace(".png", "_A_baseline.png"), "and", outB)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="a cozy reading nook with warm ambient light, 35mm film grain")
    p.add_argument("--out", type=str, default="psy_test.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=12345)

    # knobs
    p.add_argument("--act", type=str, default="silu",
                   choices=["silu","gelu","gelu_tanh","relu","leakyrelu","mish","hswish","softsign","softplus"])
    p.add_argument("--tau", type=float, default=1.0, help="Pre-activation temperature (>1 flattens curvature)")
    p.add_argument("--beta", type=float, default=0.0, help="Identity blend (0..1)")
    p.add_argument("--gamma", type=float, default=1.0, help="Blend toward new act before identity")

    # targeting
    p.add_argument("--stages", type=str, default="up,mid", help="Comma list from {down,mid,up}")
    p.add_argument("--start-idx", type=int, default=0, help="First resblock index within selected stages")
    p.add_argument("--patch-mlp", action="store_true", help="Also patch attention MLP activations")

    # calibration
    p.add_argument("--calibrate", action="store_true", help="Variance-match to baseline using one baseline pass")

    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()
