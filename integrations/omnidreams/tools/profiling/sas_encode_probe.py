#!/usr/bin/env python3
"""ncu probe for the SAS-encode (quantize) server overhead. Profiles one
q.compress of a chunk-shaped latent, bracketed for --profile-from-start off.

  ncu --profile-from-start off --metrics sm__cycles_active.sum,gpu__time_duration.sum \
    -f -o sas2_enc  python sas_encode_probe.py int4-2s
"""
import sys
import torch


def main() -> int:
    preset = sys.argv[1] if len(sys.argv) > 1 else "int4-2s"
    from flashdreams.infra.sas import build_preset_quantizer
    q = build_preset_quantizer(preset)
    lat = torch.randn(8, 16, 88, 160, device="cuda", dtype=torch.float16)  # one chunk latent
    for _ in range(3):        # warmup (k-means init, allocator)
        q.compress(lat)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    q.compress(lat)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"[sas-enc] profiled {preset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
