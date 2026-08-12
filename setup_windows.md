# Windows Setup for Flashdream Interactive-Drive

## Requirements
- Windows 11 with CUDA 13.0
- Python 3.11.15 (in `.venv`)
- Visual Studio 2022 Community
- PyTorch 2.8.x (cu130 wheels) — see [PyTorch Version](#pytorch-version) below

## Setup Steps

### 1. Run Complete Setup
```powershell
.\setup_interactive_drive.bat
```

This script:
- Syncs dependencies via **narrow `uv sync --package flashdreams-omnidreams`** (preserves your torch version)
- Downloads models (Cosmos-Reason1, LightWave VAE/TAE, OmniDreams)
- Builds C++ extensions (Ludus renderer, PhysX)
- Optional: Precompiles torch.compile cache (skipped on Windows by default)

### 2. Run Interactive-Drive
```powershell
.\run_interactive_drive_perf.bat --game-mode
```

## Controls
- **WASD** - Drive
- **Mouse** - Look around
- **C** - Spawn obstacle
- **R** - Restart session
- **Esc** - Quit

## Prompt Editing
Type in the Scene Prompt field:
- `/spawn car 30 5` - Spawn vehicle
- `/clear-actors` - Clear all actors

## Windows-Specific Notes

### PyTorch Version

**Use PyTorch 2.8.x (cu130), not 2.12.1+**

The project requires `torch>=2.9`, but PyTorch 2.12.1+ has a broken functorch integration on Windows:
```
ImportError: cannot import name 'min_cut_rematerialization_partition' from 'functorch.compile'
```
This occurs during `torch._dynamo` compiler initialization before environment variables like `TORCH_COMPILE_DISABLE` can take effect.

**Setup uses narrow sync to preserve your torch version:**
```powershell
uv sync --package flashdreams-omnidreams --extra dev --extra interactive-drive
```

This respects the workspace's dependency pins instead of upgrading to the latest (2.12.1). If you need a specific torch version:
```powershell
uv pip install "torch==2.8.1+cu130" --index https://download.pytorch.org/whl/cu130
```

### torch.compile on Windows
PyTorch has broken functorch integration on Windows (functorch.compile.min_cut_rematerialization_partition missing during compiler init). 

**Solution:** Patch `flashdreams/infra/compile.py` to skip torch.compile on Windows:

```python
def compile_module(module: M, *, mode: CompileMode = "max-autotune-no-cudagraphs") -> M:
    if sys.platform == "win32":
        return module  # Skip compilation on Windows
    _configure_inductor_cache()
    _patch_triton_bundle_collection()
    return cast(M, torch.compile(module, mode=mode))
```

This allows the app to run in eager mode on Windows (slightly slower but stable), while Linux still uses torch.compile.

**Already applied:** The patch is in the repo. If you rebuild, clear Python cache:
```powershell
Remove-Item -Recurse -Force flashdreams\flashdreams\infra\__pycache__
```

### Ludus C++ Extension
Requires MSVC compiler setup via vcvarsall.bat. The setup script calls this automatically.

If compilation fails:
```powershell
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

### Performance
- First chunk: ~14 seconds (includes model warmup)
- Subsequent chunks: ~2-3 seconds at 1168x640@30fps
- Use `--perf` flag for optimized inference

## Troubleshooting

**"No module named pip"**
The venv was created by `uv`, which doesn't include pip. Use `uv pip` instead or `uv sync` for dependency management.

**"ImportError: min_cut_rematerialization_partition"**
PyTorch 2.12.1+ functorch is broken on Windows. Use 2.8.x:
```powershell
uv pip install "torch==2.8.1+cu130" --index https://download.pytorch.org/whl/cu130
```
Then clear Python cache: `Remove-Item -Recurse -Force flashdreams\flashdreams\infra\__pycache__`

**Ludus build fails**
Check that MSVC and Windows SDK headers are installed. Run vcvarsall.bat x64 manually and retry.
