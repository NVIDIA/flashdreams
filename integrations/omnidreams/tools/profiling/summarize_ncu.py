#!/usr/bin/env python3
"""Aggregate an `ncu --csv --page raw` export into decode HW-load headline numbers."""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hdr_i = next((i for i, r in enumerate(rows)
              if any("Kernel Name" in c or c.strip() == "ID" for c in r)), 0)
hdr = [c.strip() for c in rows[hdr_i]]
data = rows[hdr_i + 1:]


def ci(sub):
    return next((i for i, h in enumerate(hdr) if sub in h), None)


def num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


i_name = ci("Kernel Name")
i_dur = ci("gpu__time_duration.sum")   # milliseconds (per the units row)
i_cyc = ci("sm__cycles_active.sum")    # active cycles summed across all SMs
i_smt = ci("sm__throughput.avg.pct")
i_dram = ci("dram__throughput.avg.pct")
i_occ = ci("warps_active.avg.pct")

kernels = []
for r in data:
    if not r or i_dur is None or i_dur >= len(r):
        continue
    d = num(r[i_dur])
    if d is None:
        continue
    kernels.append({
        "name": (r[i_name] if i_name is not None and i_name < len(r) else "?"),
        "dur": d,
        "cyc": num(r[i_cyc]) or 0.0 if i_cyc is not None else 0.0,
        "smt": num(r[i_smt]) or 0.0 if i_smt is not None else 0.0,
        "dram": num(r[i_dram]) or 0.0 if i_dram is not None else 0.0,
        "occ": num(r[i_occ]) or 0.0 if i_occ is not None else 0.0,
    })

n = len(kernels)
tot_dur = sum(k["dur"] for k in kernels)
tot_cyc = sum(k["cyc"] for k in kernels)


def wavg(key):
    return sum(k[key] * k["dur"] for k in kernels) / tot_dur if tot_dur else 0.0


print(f"kernels profiled           : {n}")
print(f"decode GPU time (sum)      : {tot_dur:.3f} ms   ({tot_dur*1e3:.0f} us)")
print(f"SM active cycles (all SMs) : {tot_cyc/1e9:.2f} G   ({tot_cyc/1e6:.0f} M)")
print(f"SM compute throughput      : {wavg('smt'):.1f} % of peak  (time-weighted)")
print(f"DRAM memory throughput     : {wavg('dram'):.1f} % of peak  (time-weighted)")
print(f"achieved occupancy         : {wavg('occ'):.1f} %")
print("top kernels by GPU time:")
for k in sorted(kernels, key=lambda x: -x["dur"])[:6]:
    print(f"  {k['dur']*1e3:8.1f} us  SM {k['smt']:4.1f}%  DRAM {k['dram']:4.1f}%  "
          f"occ {k['occ']:4.1f}%  {k['name'][:64]}")
