# Self-Forcing

Baselines:
- [Official Repo Profiling](https://github.com/liruilong940607/Self-Forcing/pull/1#issue-4396013709)

### With flashdreams env:

```bash
uv run --package flashdreams --extra examples \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    flashdreams/examples/run_causal_wan21.py \
    --config_name self_forcing_lighttae \
    --total_blocks 7
```

On A100
> AR 0 encode 0.021 ms diffuse 16255.894 ms decode 143063.156 ms finalize 98.825 ms | total(w/o finalize) 159319.070 ms total 159417.895 ms | GPU mem alloc 19.162 GiB reserved 21.787 GiB peak 21.319 GiB
> AR 1 encode 0.021 ms diffuse 1683.165 ms decode 56416.598 ms finalize 124.611 ms | total(w/o finalize) 58099.784 ms total 58224.396 ms | GPU mem alloc 19.163 GiB reserved 21.787 GiB peak 21.319 GiB
> AR 2 encode 0.021 ms diffuse 642.681 ms decode 13.634 ms finalize 150.875 ms | total(w/o finalize) 656.336 ms total 807.210 ms | GPU mem alloc 19.162 GiB reserved 21.787 GiB peak 21.319 GiB
> AR 3 encode 0.021 ms diffuse 748.495 ms decode 49.293 ms finalize 176.603 ms | total(w/o finalize) 797.809 ms total 974.412 ms | GPU mem alloc 19.189 GiB reserved 21.918 GiB peak 21.319 GiB
> AR 4 encode 0.021 ms diffuse 851.870 ms decode 13.478 ms finalize 204.958 ms | total(w/o finalize) 865.368 ms total 1070.326 ms | GPU mem alloc 19.189 GiB reserved 21.922 GiB peak 21.319 GiB
> AR 5 encode 0.022 ms diffuse 954.896 ms decode 13.474 ms finalize 228.108 ms | total(w/o finalize) 968.392 ms total 1196.499 ms | GPU mem alloc 19.189 GiB reserved 21.922 GiB peak 21.319 GiB
> AR 6 encode 0.021 ms diffuse 1057.509 ms decode 13.473 ms finalize 254.318 ms | total(w/o finalize) 1071.003 ms total 1325.321 ms | GPU mem alloc 19.189 GiB reserved 21.922 GiB peak 21.319 GiB

On GB200
> AR 0 encode 0.022 ms diffuse 47020.234 ms decode 48014.711 ms finalize 43.231 ms | total(w/o finalize) 95034.968 ms total 95078.198 ms | GPU mem alloc 19.162 GiB reserved 22.135 GiB peak 21.407 GiB
> AR 1 encode 0.026 ms diffuse 1607.381 ms decode 43041.988 ms finalize 38.407 ms | total(w/o finalize) 44649.396 ms total 44687.803 ms | GPU mem alloc 19.163 GiB reserved 22.135 GiB peak 21.407 GiB
> AR 2 encode 0.022 ms diffuse 214.317 ms decode 4.320 ms finalize 35.907 ms | total(w/o finalize) 218.660 ms total 254.566 ms | GPU mem alloc 19.162 GiB reserved 22.135 GiB peak 21.407 GiB
> AR 3 encode 0.022 ms diffuse 198.750 ms decode 89.716 ms finalize 40.423 ms | total(w/o finalize) 288.487 ms total 328.911 ms | GPU mem alloc 19.197 GiB reserved 21.533 GiB peak 21.407 GiB
> AR 4 encode 0.023 ms diffuse 209.456 ms decode 3.531 ms finalize 36.530 ms | total(w/o finalize) 213.011 ms total 249.541 ms | GPU mem alloc 19.197 GiB reserved 21.537 GiB peak 21.407 GiB
> AR 5 encode 0.021 ms diffuse 202.850 ms decode 3.532 ms finalize 37.491 ms | total(w/o finalize) 206.403 ms total 243.894 ms | GPU mem alloc 19.197 GiB reserved 21.537 GiB peak 21.407 GiB
> AR 6 encode 0.023 ms diffuse 213.910 ms decode 3.537 ms finalize 40.247 ms | total(w/o finalize) 217.470 ms total 257.717 ms | GPU mem alloc 19.197 GiB reserved 21.537 GiB peak 21.407 GiB

On H100
> AR 0 encode 0.020 ms diffuse 10691.640 ms decode 104422.930 ms finalize 67.661 ms | total(w/o finalize) 115114.590 ms total 115182.250 ms | GPU mem alloc 19.186 GiB reserved 21.756 GiB peak 21.352 GiB
> AR 1 encode 0.022 ms diffuse 1575.072 ms decode 37327.184 ms finalize 41.953 ms | total(w/o finalize) 38902.278 ms total 38944.231 ms | GPU mem alloc 19.187 GiB reserved 21.756 GiB peak 21.352 GiB
> AR 2 encode 0.022 ms diffuse 251.339 ms decode 6.750 ms finalize 50.078 ms | total(w/o finalize) 258.111 ms total 308.189 ms | GPU mem alloc 19.186 GiB reserved 21.756 GiB peak 21.352 GiB
> AR 3 encode 0.015 ms diffuse 277.743 ms decode 33.834 ms finalize 55.888 ms | total(w/o finalize) 311.591 ms total 367.478 ms | GPU mem alloc 19.245 GiB reserved 21.887 GiB peak 21.352 GiB
> AR 4 encode 0.015 ms diffuse 311.966 ms decode 6.676 ms finalize 65.200 ms | total(w/o finalize) 318.658 ms total 383.858 ms | GPU mem alloc 19.244 GiB reserved 21.891 GiB peak 21.352 GiB
> AR 5 encode 0.016 ms diffuse 343.706 ms decode 6.646 ms finalize 74.226 ms | total(w/o finalize) 350.367 ms total 424.593 ms | GPU mem alloc 19.245 GiB reserved 21.891 GiB peak 21.352 GiB
> AR 6 encode 0.020 ms diffuse 375.887 ms decode 6.678 ms finalize 80.734 ms | total(w/o finalize) 382.586 ms total 463.319 ms | GPU mem alloc 19.244 GiB reserved 21.891 GiB peak 21.352 GiB

### With self-forcing env:

```bash
conda activate self_forcing
pip install mediapy pynvml loguru boto3 transformer-engine[pytorch,core-cu12]
PYTHONPATH=./flashdreams python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  flashdreams/examples/run_causal_wan21.py \
  --config_name self_forcing_lighttae \
  --total_blocks 7
```

On A100
> AR 0 encode 0.021 ms diffuse 9719.179 ms decode 4394.018 ms finalize 97.804 ms | total(w/o finalize) 14113.218 ms total 14211.022 ms | GPU mem alloc 19.078 GiB reserved 21.289 GiB peak 20.197 GiB
> AR 1 encode 0.023 ms diffuse 14469.375 ms decode 11246.406 ms finalize 122.569 ms | total(w/o finalize) 25715.805 ms total 25838.374 ms | GPU mem alloc 19.079 GiB reserved 21.289 GiB peak 20.224 GiB
> AR 2 encode 0.023 ms diffuse 636.438 ms decode 14.221 ms finalize 148.522 ms | total(w/o finalize) 650.681 ms total 799.204 ms | GPU mem alloc 19.078 GiB reserved 21.289 GiB peak 20.224 GiB
> AR 3 encode 0.022 ms diffuse 741.092 ms decode 469.722 ms finalize 175.246 ms | total(w/o finalize) 1210.837 ms total 1386.083 ms | GPU mem alloc 19.113 GiB reserved 21.402 GiB peak 20.232 GiB
> AR 4 encode 0.022 ms diffuse 840.731 ms decode 14.098 ms finalize 201.189 ms | total(w/o finalize) 854.851 ms total 1056.040 ms | GPU mem alloc 19.113 GiB reserved 21.404 GiB peak 20.232 GiB
>  AR 5 encode 0.021 ms diffuse 944.683 ms decode 14.080 ms finalize 226.367 ms | total(w/o finalize) 958.784 ms total 1185.151 ms | GPU mem alloc 19.113 GiB reserved 21.404 GiB peak 20.232 GiB
> AR 6 encode 0.022 ms diffuse 1048.089 ms decode 14.090 ms finalize 253.412 ms | total(w/o finalize) 1062.201 ms total 1315.613 ms | GPU mem alloc 19.113 GiB reserved 21.404 GiB peak 20.232 GiB
