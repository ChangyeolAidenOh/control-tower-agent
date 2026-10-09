"""Stage 3-C gate: Chronos-2 small availability check on M2 (ST3-03 scope).

Verifies, before any 3-C modeling code is written:
  1. package import (chronos-forecasting >= 2.0) and model download
     (autogluon/chronos-2-small, ~112 MB safetensors, apache-2.0,
     ungated -- no paid element; stop and report if anything prompts
     for credentials or payment),
  2. CPU load time, parameter count, tiny-inference latency,
  3. MPS availability and, if present, the same tiny inference on MPS,
  4. which prediction API the installed version exposes (introspected,
     not assumed) and whether quantile output works.

Run: python -m scripts.check_chronos
Install first: pip install 'chronos-forecasting>=2.0'
"""

import resource
import sys
import time

import numpy as np

MODEL_ID = "autogluon/chronos-2-small"
HORIZON = 28
QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95]


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def tiny_inference(pipeline, device_label: str) -> None:
    rng = np.random.default_rng(0)
    context = [rng.poisson(3.0, 200).astype(np.float32) for _ in range(4)]
    t0 = time.time()
    if hasattr(pipeline, "predict_quantiles"):
        out = pipeline.predict_quantiles(
            context, prediction_length=HORIZON, quantile_levels=QUANTILES)
        api = "predict_quantiles"
    else:
        out = pipeline.predict(context, prediction_length=HORIZON)
        api = "predict"
    shapes = [getattr(o, "shape", None) for o in
              (out if isinstance(out, (list, tuple)) else [out])]
    print(f"{device_label} api={api} t_s={time.time() - t0:.1f} "
          f"out_shapes={shapes} rss_mb={rss_mb():.0f}")


def main() -> None:
    import chronos
    print(f"chronos-forecasting version={getattr(chronos, '__version__', '?')}")
    from chronos import Chronos2Pipeline

    t0 = time.time()
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cpu")
    print(f"loaded {MODEL_ID} cpu load_s={time.time() - t0:.1f} "
          f"rss_mb={rss_mb():.0f}")
    members = [m for m in dir(pipeline)
               if "predict" in m and not m.startswith("_")]
    print(f"predict members={members}")
    tiny_inference(pipeline, "cpu")

    try:
        import torch
        mps = torch.backends.mps.is_available()
        print(f"mps_available={mps}")
        if mps:
            t0 = time.time()
            pipeline_mps = Chronos2Pipeline.from_pretrained(
                MODEL_ID, device_map="mps")
            print(f"mps load_s={time.time() - t0:.1f}")
            tiny_inference(pipeline_mps, "mps")
    except Exception as exc:
        print(f"mps probe failed (cpu path remains valid): {exc!r}")

    print("chronos check PASS")


if __name__ == "__main__":
    main()
