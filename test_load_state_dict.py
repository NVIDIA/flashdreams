#!/usr/bin/env python3
"""Minimal test of load_state_dict hang - no Ludus/rasterizer required."""
import os
os.environ['TORCH_COMPILE_DEBUG'] = '0'
import sys
import time
sys.path.insert(0, 'integrations/omnidreams')

start = time.time()

def log(msg):
    elapsed = time.time() - start
    print(f'[{elapsed:7.2f}s] {msg}', flush=True)

log('[TEST] PyTorch version:')
import torch
log(f'  torch {torch.__version__}')
log(f'  CUDA available: {torch.cuda.is_available()}')

log('[TEST] Loading omnidreams model...')
try:
    from omnidreams.pipeline import OmnidreamsPipelineConfig
    from flashdreams.infra.config import derive_config

    # Use the perf config
    log('[TEST] Creating OmnidreamsPipelineConfig...')
    from omnidreams.config import SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
    config = SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE

    log('[TEST] Deriving pipeline config...')
    pipeline_config = derive_config(config)

    log('[TEST] Disabling torch.compile on Windows...')
    if sys.platform == "win32":
        # Disable all compilation
        pipeline_config.diffusion_model.transformer.compile_network = False
        if hasattr(pipeline_config, 'decoder') and pipeline_config.decoder:
            pipeline_config.decoder.compile_network = False
        log('[TEST] torch.compile disabled globally')

    log('[TEST] Building pipeline...')
    pipeline = pipeline_config.setup().to(device=torch.device('cuda:0'))

    log('[TEST] ✓ Model loaded successfully')
    log(f'[TEST] Pipeline type: {type(pipeline).__name__}')

except Exception as e:
    log(f'[TEST] ERROR: {type(e).__name__}: {str(e)[:200]}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

log('[TEST] ✓ Test complete')
