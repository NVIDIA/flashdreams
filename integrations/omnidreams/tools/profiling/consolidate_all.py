#!/usr/bin/env python3
"""Consolidate all measured data into story_table_consolidated.xlsx.

Inputs (on the box):
  SW      : /scratch/nisjain/sw_run/sw_per_chunk.csv          (offline, per-chunk)
  Payload : /scratch/nisjain/chunk_run_1000/per_chunk.csv     (offline setup)
  Quality : same file (psnr_* columns)
  HW      : ncu import CSVs (per-kernel; grouped per NVTX chunk):
              /scratch/nisjain/hw_run/token5_import.csv
              /scratch/nisjain/hw_run/pixel5_import.csv
            SAS-encode ncu import CSVs (summed over all kernels):
              /scratch/nisjain/hw_run/sas2_enc_import.csv
              /scratch/nisjain/hw_run/sas4_enc_import.csv

Full precision: SM-cycle counts written as exact integers (Excel exact <15 digits);
also kept as text strings so no unit place is ever lost. Times/fps keep decimals.
"""
import csv
import io
import os
import re
import sys

import pandas as pd
from openpyxl.utils import get_column_letter

BOX = "/scratch/nisjain"
FPC = 8                      # frames per chunk
NVENC_MS = 5.595            # live-measured NVENC per chunk (pixel before-network add-on)


# ---------- ncu CSV parsing (per-chunk SM cycles) ----------
def _load_ncu(path):
    raw = open(path, newline="", encoding="utf-8", errors="replace").read().splitlines()
    start = next((i for i, ln in enumerate(raw) if ln.startswith('"') and ln.count(",") >= 3), 0)
    return list(csv.DictReader(io.StringIO("\n".join(raw[start:]))))


def _num(v):
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s in ("", "n/a", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def ncu_per_chunk(path):
    """return (dict chunk->cycles, total, unattributed)."""
    rows = _load_ncu(path)
    if not rows:
        return {}, 0, 0
    cols = list(rows[0].keys())
    nvtx = next((c for c in cols if "push/pop" in c.lower()), None) \
        or next((c for c in cols if "nvtx" in c.lower() or "range" in c.lower()), None)
    direct = next((c for c in cols if "sm__cycles_active.sum" in c.lower()), None)
    namec = next((c for c in cols if c.strip().lower() == "metric name"), None)
    valc = next((c for c in cols if c.strip().lower() == "metric value"), None)
    per, total, un = {}, 0.0, 0.0
    for r in rows:
        if direct:
            v = _num(r.get(direct))
        elif namec and r.get(namec, "").strip() == "sm__cycles_active.sum":
            v = _num(r.get(valc))
        else:
            v = None
        if v is None:
            continue
        total += v
        ch = None
        if nvtx and r.get(nvtx):
            m = re.search(r"chunk(\d+)", r[nvtx])
            if m:
                ch = int(m.group(1))
        if ch is None:
            un += v
        else:
            per[ch] = per.get(ch, 0.0) + v
    return {k: int(round(v)) for k, v in per.items()}, int(round(total)), int(round(un))


def ncu_total(path):
    """sum sm__cycles_active over all kernels (for the SAS-encode op, no NVTX split)."""
    _, total, _ = ncu_per_chunk(path)
    return total


def main():
    out_xlsx = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BOX, "story_table_consolidated.xlsx")

    # ---------- SW (offline, per-chunk) ----------
    sw = pd.read_csv(os.path.join(BOX, "sw_run/sw_per_chunk.csv"))
    sw["pixel_before_ms"] = sw["pixel_gen_ms"] + NVENC_MS
    sw["token_before_ms"] = sw["token_gen_ms"]
    sw["sas2_before_ms"] = sw["token_gen_ms"] + sw["sas2_ms"]
    sw["sas4_before_ms"] = sw["token_gen_ms"] + sw["sas4_ms"]
    for m in ("pixel", "token", "sas2", "sas4"):
        sw[f"{m}_fps"] = FPC / (sw[f"{m}_before_ms"] / 1000.0)
        # explicit chunk-level markers T0..Tn = cumulative before-network time (ms)
        sw[f"{m}_T_ms"] = sw[f"{m}_before_ms"].cumsum()
        sw[f"{m}_fps_cum"] = FPC * (sw.index + 1) / (sw[f"{m}_T_ms"] / 1000.0)
    sw = sw.round(6)

    # ---------- Payload + Quality (offline setup) ----------
    pq = pd.read_csv(os.path.join(BOX, "chunk_run_1000/per_chunk.csv"))
    payload = pq[["chunk", "raw_bytes", "sas2_bytes", "sas4_bytes", "h264_bytes"]].copy()
    for c in ("raw", "sas2", "sas4", "h264"):
        payload[f"{c}_KB"] = pd.to_numeric(payload[f"{c}_bytes"], errors="coerce") / 1024.0
    payload = payload.round(4)
    quality = pq[["chunk", "psnr_pixel_h264", "psnr_sas2_vs_raw", "psnr_sas4_vs_raw"]].copy()

    # ---------- HW SM-cycles (ncu, 5 chunks) ----------
    hw_dir = os.path.join(BOX, "hw_run")
    hw_rows = []
    tok_per, dec_per, pix_per = {}, {}, {}
    tok_tot = None
    # token/SAS generate: prefer the 5-chunk run, fall back to the 3-chunk run
    tp = next((os.path.join(hw_dir, f) for f in ("token5_import.csv", "token3_import.csv")
               if os.path.exists(os.path.join(hw_dir, f))), None)
    if tp:
        tok_per, tok_tot, _ = ncu_per_chunk(tp)
    # VAE-decode delta (pixel = token + decode)
    dp = os.path.join(hw_dir, "decode4_import.csv")
    if os.path.exists(dp):
        dec_per, dec_tot, _ = ncu_per_chunk(dp)
    # pixel generate = token generate + VAE decode, per chunk where both exist
    if dec_per and tok_per:
        dmean = int(round(sum(dec_per.values()) / len(dec_per)))
        for c in tok_per:
            pix_per[c] = tok_per[c] + dec_per.get(c, dmean)
    for c in sorted(tok_per):
        hw_rows.append({"chunk": c,
                        "token_generate_cycles": tok_per.get(c),
                        "vae_decode_cycles": dec_per.get(c),
                        "pixel_generate_cycles": pix_per.get(c)})
    hw = pd.DataFrame(hw_rows)

    sas2_enc = sas4_enc = None
    s2 = os.path.join(hw_dir, "sas2_enc_import.csv")
    s4 = os.path.join(hw_dir, "sas4_enc_import.csv")
    if os.path.exists(s2):
        sas2_enc = ncu_total(s2)
    if os.path.exists(s4):
        sas4_enc = ncu_total(s4)

    def _mean(d):
        return int(round(sum(d.values()) / len(d))) if d else None
    def _sum(d):
        return int(sum(d.values())) if d else None
    hw_summary = pd.DataFrame({
        "region": ["token/SAS generate (per chunk, mean)",
                   "VAE decode (per chunk, mean)",
                   "pixel generate = token + decode (per chunk, mean)",
                   "token/SAS generate (collective, all profiled chunks)",
                   "pixel generate (collective)",
                   "SAS int4-2s encode (per chunk)", "SAS int4-4s encode (per chunk)"],
        "sm_active_cycles": [_mean(tok_per), _mean(dec_per), _mean(pix_per),
                             tok_tot if tok_tot else _sum(tok_per), _sum(pix_per),
                             sas2_enc, sas4_enc],
    })
    hw_summary["sm_active_cycles_text"] = hw_summary["sm_active_cycles"].apply(
        lambda v: f"{v:,}" if v is not None else "")

    # ---------- Summary ----------
    st = sw.iloc[20:] if len(sw) > 40 else sw
    def _m(col): return round(float(st[col].mean()), 6)
    summ = pd.DataFrame({
        "metric": ["pixel throughput (fps)", "token throughput (fps)",
                   "SAS int4-2s throughput (fps)", "SAS int4-4s throughput (fps)",
                   "pixel before-network (ms/chunk)", "token before-network (ms/chunk)",
                   "SAS-2s before-network (ms/chunk)", "SAS-4s before-network (ms/chunk)",
                   "raw payload (KB/chunk)", "SAS-2s payload (KB/chunk)",
                   "SAS-4s payload (KB/chunk)", "H.264 payload (KB/chunk)",
                   "PSNR pixel vs H.264 (dB)", "PSNR SAS-2s vs raw (dB)", "PSNR SAS-4s vs raw (dB)"],
        "value": [_m("pixel_fps"), _m("token_fps"), _m("sas2_fps"), _m("sas4_fps"),
                  _m("pixel_before_ms"), _m("token_before_ms"), _m("sas2_before_ms"), _m("sas4_before_ms"),
                  round(payload["raw_KB"].mean(), 4), round(payload["sas2_KB"].mean(), 4),
                  round(payload["sas4_KB"].mean(), 4), round(payload["h264_KB"].mean(), 4),
                  round(quality["psnr_pixel_h264"].mean(), 4),
                  round(quality["psnr_sas2_vs_raw"].mean(), 4),
                  round(quality["psnr_sas4_vs_raw"].mean(), 4)],
    })

    prov = pd.DataFrame({
        "item": ["SW throughput", "HW SM-cycles", "Payload", "Quality", "precision note"],
        "source / method": [
            "OFFLINE driving (standalone generate loop, graphs ON). before-network = "
            "gen + encode, excl. ack-wait. pixel adds live-measured NVENC 5.595 ms.",
            "Nsight Compute, 5 chunks, graphs OFF, kernel replay, sm__cycles_active.sum "
            "summed per NVTX chunk range. token=decode-skipped generate (=SAS generate); "
            "pixel=decode-True generate. SAS-encode = the quantize op alone.",
            "OFFLINE setup (chunk_run_1000): raw fp16 / SAS int4-2s / int4-4s / H.264 bytes.",
            "OFFLINE setup: PSNR pixel = before vs after libx264; SAS = raw-token decode vs SAS decode.",
            "SM-cycle counts are exact integers (also given as text). Times to 6 decimals.",
        ],
    })

    # ---------- write ----------
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        sw.to_excel(xw, sheet_name="SW_throughput", index=False)
        hw.to_excel(xw, sheet_name="HW_SMcycles_perchunk", index=False)
        hw_summary.to_excel(xw, sheet_name="HW_SMcycles_summary", index=False)
        payload.to_excel(xw, sheet_name="Payload", index=False)
        quality.to_excel(xw, sheet_name="Quality", index=False)
        summ.to_excel(xw, sheet_name="Summary", index=False)
        prov.to_excel(xw, sheet_name="Provenance", index=False)
        # force plain-integer format on cycle columns (no scientific / no rounding display)
        for sh, cols in [("HW_SMcycles_perchunk", ["token_generate_cycles", "vae_decode_cycles", "pixel_generate_cycles"]),
                         ("HW_SMcycles_summary", ["sm_active_cycles"])]:
            ws = xw.sheets[sh]
            header = [c.value for c in ws[1]]
            for cn in cols:
                if cn in header:
                    ci = header.index(cn) + 1
                    for r in range(2, ws.max_row + 1):
                        ws.cell(row=r, column=ci).number_format = "#,##0"

    print("wrote", out_xlsx)
    print("\nSummary:\n", summ.to_string(index=False))
    print("\nHW:\n", hw_summary[["region", "sm_active_cycles_text"]].to_string(index=False))


if __name__ == "__main__":
    main()
