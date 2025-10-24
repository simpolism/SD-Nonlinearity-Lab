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

import argparse, json, math, os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
from PIL import Image, ImageChops

try:
    from diffusers import StableDiffusionXLPipeline
except ImportError:
    StableDiffusionXLPipeline = None  # type: ignore

try:
    from diffusers import StableDiffusion3Pipeline
except ImportError:
    StableDiffusion3Pipeline = None  # type: ignore

torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"

_PIPELINE_LOOKUP: Dict[str, Optional[Any]] = {
    "sd": StableDiffusionPipeline,
    "sdxl": StableDiffusionXLPipeline,
    "sd3": StableDiffusion3Pipeline,
}

# ---------------------- Pipeline Helper ----------------------

def resolve_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        mapping = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        key = dtype.lower()
        if key not in mapping:
            raise ValueError(f"Unsupported dtype string '{dtype}'. Use one of {list(mapping.keys())}.")
        return mapping[key]
    raise TypeError(f"Unsupported dtype specifier: {dtype!r}")

def load_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    pipeline_kind: str = "auto",
    dtype: torch.dtype = torch.float16,
    device_override: str = device,
    *,
    attention_slicing: bool = True,
    vae_tiling: bool = False,
    vae_slicing: bool = False,
    cpu_offload: bool = False,
    **pipe_kwargs: Any,
) -> Any:
    """
    Load a Stable Diffusion-family pipeline onto the requested device. Intended for reuse
    across multiple generation calls (e.g., parameter sweeps). Supports SD v1/v2, SDXL,
    and SD3 via either specific pipeline classes or AutoPipelineForText2Image.
    """
    pipeline_kind = (pipeline_kind or "auto").lower()
    if pipeline_kind == "auto":
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            **pipe_kwargs
        )
    else:
        pipe_cls = _PIPELINE_LOOKUP.get(pipeline_kind)
        if pipe_cls is None:
            raise ValueError(f"Pipeline kind '{pipeline_kind}' is not available in this environment.")
        pipe = pipe_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            **pipe_kwargs
        )

    # Disable safety checker for experimentation (user requested).
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False

    if cpu_offload:
        offload_enabled = False
        try:
            pipe.enable_model_cpu_offload()
            offload_enabled = True
        except Exception:
            try:
                pipe.enable_sequential_cpu_offload()
                offload_enabled = True
            except Exception:
                pass
        if attention_slicing and not offload_enabled:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
    else:
        pipe = pipe.to(device_override)
        if attention_slicing:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

    if vae_slicing:
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    if vae_tiling:
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
    if attention_slicing and cpu_offload:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    return pipe

@dataclass
class BaselineRecord:
    image: Image.Image
    path: str
    store: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]
    calibrate: bool

def resolve_denoiser(pipe: Any, preferred: Optional[str] = None) -> Tuple[nn.Module, str]:
    """
    Find the denoising network inside the pipeline (UNet for SD 1.x/XL, transformer for SD3, etc.).
    """
    search_order = []
    if preferred:
        search_order.append(preferred)
    search_order.extend(["unet", "transformer", "denoiser", "image_unet"])

    for attr in search_order:
        if not attr:
            continue
        if hasattr(pipe, attr):
            module = getattr(pipe, attr)
            if isinstance(module, nn.Module):
                return module, attr
    raise ValueError("Could not locate a denoiser module on the provided pipeline.")

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
    if name.startswith("down_blocks.") or ".down_blocks." in name: return "down"
    if name.startswith("up_blocks.")   or ".up_blocks."   in name: return "up"
    if name.startswith("mid_block.")   or ".mid_block."   in name: return "mid"
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

def patch_denoiser(denoiser: nn.Module, cfg: PatchCfg) -> Tuple[int, Dict[str, PsyAct], List[Tuple[nn.Module, str, nn.Module]]]:
    from diffusers.models.resnet import ResnetBlock2D
    count = 0
    patched: Dict[str, PsyAct] = {}
    idx_by_stage = {"down": -1, "mid": -1, "up": -1}
    replacements: List[Tuple[nn.Module, str, nn.Module]] = []
    patched_fullnames = set()

    def stage_allowed(stage: Optional[str]) -> bool:
        if "all" in cfg.stages:
            return True
        return stage in cfg.stages

    # Replace SiLU inside ResnetBlock2D (core UNet nonlinearity)
    for fullname, m in denoiser.named_modules():
        st = stage_of_name(fullname)
        if isinstance(m, ResnetBlock2D) and stage_allowed(st):
            stage = st or "all"
            if stage in idx_by_stage:
                idx_by_stage[stage] += 1
                if stage in cfg.stages and idx_by_stage[stage] < cfg.start_idx:
                    continue
            for attr, child in list(m._modules.items()):
                if isinstance(child, nn.SiLU):
                    psy = build_psy_factory(cfg.act_kind, cfg.tau, cfg.beta, cfg.gamma)()
                    m._modules[attr] = psy
                    fullname_attr = f"{fullname}.{attr}" if fullname else attr
                    patched[fullname_attr] = psy
                    replacements.append((m, attr, child))
                    patched_fullnames.add(fullname_attr)
                    count += 1

    # General fallback: patch remaining leaf SiLU/GELU modules within allowed scopes
    for parent, attr, child, fullname in _iter_leaf_acts(denoiser):
        st = stage_of_name(fullname)
        if not stage_allowed(st):
            continue
        fullname_attr = fullname
        if fullname_attr in patched_fullnames:
            continue
        if isinstance(child, (nn.GELU, nn.SiLU)):
            psy = build_psy_factory(cfg.act_kind, cfg.tau, cfg.beta, cfg.gamma)()
            parent._modules[attr] = psy
            patched[fullname_attr] = psy
            replacements.append((parent, attr, child))
            patched_fullnames.add(fullname_attr)
            count += 1

    # Optionally patch attention MLP activations (often GELU/SiLU)
    if cfg.patch_attn_mlp:
        for parent, attr, child, fullname in _iter_leaf_acts(denoiser):
            if fullname in patched_fullnames:
                continue
            st = stage_of_name(fullname)
            if not stage_allowed(st):
                continue
            if isinstance(child, (nn.GELU, nn.SiLU)):
                psy = build_psy_factory(cfg.act_kind, cfg.tau, cfg.beta, cfg.gamma)()
                parent._modules[attr] = psy
                patched[fullname] = psy
                replacements.append((parent, attr, child))
                patched_fullnames.add(fullname)
                count += 1

    return count, patched, replacements

def restore_denoiser(replacements: List[Tuple[nn.Module, str, nn.Module]]):
    """Restore modules swapped by patch_denoiser back to their original instances."""
    for parent, attr, original in replacements:
        parent._modules[attr] = original

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
        x_cpu, y_old_cpu = records[name]
        device = psy.gain_buf.device
        x = x_cpu.to(device)
        y_old = y_old_cpu.to(device)
        y_new = psy.forward(x)
        mu_old, sd_old = y_old.mean().item(), y_old.std().item()
        mu_new, sd_new = y_new.mean().item(), y_new.std().item()
        g = (sd_old / max(sd_new, eps))
        b = (mu_old - g * mu_new)
        psy.gain_buf.data.fill_(g)
        psy.bias_buf.data.fill_(b)
        del x, y_old, y_new

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
            self.store[self.key] = (x.detach().cpu(), y.detach().cpu())
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

def _build_pipe_call_kwargs(args, generator: torch.Generator) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.cfg,
        "generator": generator,
    }
    if getattr(args, "negative_prompt", None):
        kwargs["negative_prompt"] = args.negative_prompt
    if getattr(args, "height", None):
        kwargs["height"] = args.height
    if getattr(args, "width", None):
        kwargs["width"] = args.width
    return kwargs

# ---------------------- Main Run ----------------------

def run(
    args,
    pipe: Any = None,
    baseline_cache: Optional[Dict[Any, BaselineRecord]] = None,
    baseline_key: Optional[Any] = None,
    baseline_path: Optional[str] = None,
    save_baseline: bool = True,
):
    own_pipe = False
    skip_baseline = getattr(args, "skip_baseline", False)
    if skip_baseline and getattr(args, "calibrate", False):
        print("[Skip] skip_baseline=True disables calibration; ignoring --calibrate for this run.")
        args.calibrate = False
    if pipe is None:
        dtype = resolve_dtype(getattr(args, "dtype", torch.float16))
        pipe_kwargs = getattr(args, "pipe_kwargs", {}) or {}
        if isinstance(pipe_kwargs, str):
            pipe_kwargs = json.loads(pipe_kwargs)
        pipe = load_pipeline(
            getattr(args, "model_id", "runwayml/stable-diffusion-v1-5"),
            pipeline_kind=getattr(args, "pipeline", "auto"),
            dtype=dtype,
            device_override=device,
            attention_slicing=not getattr(args, "disable_attention_slicing", False),
            vae_tiling=getattr(args, "vae_tiling", False),
            vae_slicing=getattr(args, "vae_slicing", False),
            cpu_offload=getattr(args, "cpu_offload", False),
            **pipe_kwargs,
        )
        own_pipe = True

    denoiser, denoiser_attr = resolve_denoiser(pipe, getattr(args, "denoiser", None))

    cache = baseline_cache
    record: Optional[BaselineRecord] = None
    if cache is not None and baseline_key is not None:
        record = cache.get(baseline_key)

    needs_calibrate = bool(args.calibrate)
    need_new_baseline = record is None or (needs_calibrate and not record.calibrate)
    if skip_baseline:
        need_new_baseline = False
    elif not need_new_baseline and needs_calibrate:
        store_ok = isinstance(record.store, dict) and len(record.store) > 0
        if not store_ok:
            need_new_baseline = True
            if cache is not None and baseline_key is not None:
                cache.pop(baseline_key, None)
            record = None
    baseline_store: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    baseline_img: Optional[Image.Image]
    baseline_actual_path = baseline_path or args.out.replace(".png", "_A_baseline.png")

    if skip_baseline:
        baseline_img = None
        baseline_store = {}
    elif need_new_baseline:
        taps_inserted = False
        try:
            if needs_calibrate:
                baseline_store = insert_taps_for_baseline(denoiser)
                taps_inserted = True
            gen = torch.Generator(device=device).manual_seed(args.seed)
            call_kwargs = _build_pipe_call_kwargs(args, gen)
            baseline_img = pipe(**call_kwargs).images[0]
            directory = os.path.dirname(baseline_actual_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            baseline_img.save(baseline_actual_path)
        finally:
            if needs_calibrate and taps_inserted:
                remove_taps(denoiser)
        if not needs_calibrate:
            baseline_store = {}
        record = BaselineRecord(
            image=baseline_img,
            path=baseline_actual_path,
            store=baseline_store,
            calibrate=needs_calibrate,
        )
        if cache is not None and baseline_key is not None:
            cache[baseline_key] = record
    elif record is not None:
        baseline_img = record.image
        baseline_store = record.store if isinstance(record.store, dict) else {}
        baseline_actual_path = baseline_path or record.path

    # ---- Baseline (and optional stat collection) ----
    # baseline already computed/retrieved above

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
    replacements: List[Tuple[nn.Module, str, nn.Module]] = []
    n = 0
    per_act: Dict[str, PsyAct] = {}
    try:
        n, per_act, replacements = patch_denoiser(denoiser, cfg)
        print(f"[UNet] Patched {n} activation modules | act={args.act} tau={args.tau} beta={args.beta} gamma={args.gamma} | stages={cfg.stages} start_idx={cfg.start_idx} MLP={cfg.patch_attn_mlp} | denoiser={denoiser_attr}")

        # ---- Variance-match (module path only) ----
        if args.calibrate and per_act:
            print("[Calib] Performing variance matching on patched modules...")
            variance_match(per_act, baseline_store)

        # ---- Patched image with the SAME seed ----
        g = torch.Generator(device=device).manual_seed(args.seed)
        if n == 0:
            print("[Fallback] No nn.SiLU/nn.GELU modules found; patching F.silu/F.gelu functionally.")
            with patch_functional_acts(args.act, args.tau, args.beta, args.gamma):
                call_kwargs = _build_pipe_call_kwargs(args, g)
                imgB = pipe(**call_kwargs).images[0]
        else:
            call_kwargs = _build_pipe_call_kwargs(args, g)
            imgB = pipe(**call_kwargs).images[0]
    finally:
        if replacements:
            restore_denoiser(replacements)
        if own_pipe:
            try:
                pipe.to("cpu")
            except Exception:
                pass

    outA = baseline_actual_path
    outB = args.out.replace(".png", "_B_patched.png")
    directory = os.path.dirname(outB)
    if directory:
        os.makedirs(directory, exist_ok=True)
    imgB.save(outB)

    # ---- Sanity check: warn if unchanged ----
    if baseline_img is not None and ImageChops.difference(baseline_img, imgB).getbbox() is None:
        print("[WARN] Patched image is pixel-identical to baseline. "
              "If you expected differences, increase --tau / --beta or verify patching was applied.")

    if skip_baseline:
        print("Done. Wrote patched image:", outB)
    else:
        print("Done. Wrote:", outA, "and", outB)
    return {
        "baseline_path": outA if not skip_baseline else "",
        "patched_path": outB,
        "patched_modules": n,
        "used_fallback": (n == 0),
        "denoiser": denoiser_attr,
        "reuse_baseline": False if skip_baseline else not need_new_baseline,
        "skip_baseline": skip_baseline,
    }

def run_from_kwargs(
    pipe: Any = None,
    baseline_cache: Optional[Dict[Any, BaselineRecord]] = None,
    baseline_key: Optional[Any] = None,
    baseline_path: Optional[str] = None,
    save_baseline: bool = True,
    **kwargs,
):
    """
    Convenience helper so other scripts (e.g., sweeps) can call into this module without
    spawning a new Python process or reparsing CLI arguments.
    """
    defaults = {
        "prompt": "a cozy reading nook with warm ambient light, 35mm film grain",
        "out": "psy_test.png",
        "steps": 30,
        "cfg": 7.5,
        "seed": 12345,
        "act": "silu",
        "tau": 1.0,
        "beta": 0.0,
        "gamma": 1.0,
        "stages": "up,mid",
        "start_idx": 0,
        "patch_mlp": False,
        "calibrate": False,
        "model_id": "runwayml/stable-diffusion-v1-5",
        "pipeline": "auto",
        "dtype": "fp16",
        "height": None,
        "width": None,
        "negative_prompt": None,
        "denoiser": None,
        "pipe_kwargs": {},
        "vae_tiling": False,
        "vae_slicing": False,
        "cpu_offload": False,
        "disable_attention_slicing": False,
        "skip_baseline": False,
    }
    merged = {**defaults, **kwargs}
    if "prompt" not in merged or "out" not in merged:
        raise ValueError("run_from_kwargs requires at least 'prompt' and 'out'.")
    args = argparse.Namespace(**merged)
    return run(
        args,
        pipe=pipe,
        baseline_cache=baseline_cache,
        baseline_key=baseline_key,
        baseline_path=baseline_path,
        save_baseline=save_baseline,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="a cozy reading nook with warm ambient light, 35mm film grain")
    p.add_argument("--out", type=str, default="psy_test.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--height", type=int, default=None, help="Optional image height override")
    p.add_argument("--width", type=int, default=None, help="Optional image width override")
    p.add_argument("--negative-prompt", type=str, default=None, help="Optional negative prompt")

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

    # pipeline selection
    p.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="Diffusers model repo or local path")
    p.add_argument("--pipeline", type=str, default="auto", choices=["auto","sd","sdxl","sd3"], help="Pipeline loader to use")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32"], help="Computation dtype for the pipeline")
    p.add_argument("--denoiser", type=str, default=None, help="Override attribute name for the denoiser module (e.g., unet, transformer)")
    p.add_argument("--pipe-kwargs", type=str, default="{}", help="JSON dict of kwargs forwarded to pipeline.from_pretrained")
    p.add_argument("--vae-tiling", action="store_true", help="Enable VAE tiling to reduce memory usage")
    p.add_argument("--vae-slicing", action="store_true", help="Enable VAE slicing to reduce memory usage")
    p.add_argument("--cpu-offload", action="store_true", help="Enable model CPU offload (requires accelerate)")
    p.add_argument("--disable-attention-slicing", action="store_true", help="Disable attention slicing")
    p.add_argument("--skip-baseline", action="store_true", help="Skip baseline inference pass (disables calibration)")

    args = p.parse_args()
    try:
        args.pipe_kwargs = json.loads(args.pipe_kwargs) if args.pipe_kwargs else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for --pipe-kwargs: {exc}") from exc
    run(args)

if __name__ == "__main__":
    main()
