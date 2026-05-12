"""Skippy task for preprocessing AV2 tar scenes into per-file .pt cache."""

import dbm
import traceback
from dataclasses import dataclass
from pathlib import Path

from skippy.task import Task

_DBM_CACHE: dict = {}


def _lookup_manifest(manifest_path: str, key: str) -> str:
    """O(1) lookup in a dbm manifest. Handle is cached per-process."""
    if manifest_path not in _DBM_CACHE:
        _DBM_CACHE[manifest_path] = dbm.open(manifest_path, "r")
    return _DBM_CACHE[manifest_path][key].decode()


@dataclass
class PreprocessSceneCacheTask(Task):
    """Load an AV2 scene from tar, serialize to per-file .pt cache.

    Items are scene_key strings (16-char hex hashes of tar paths).
    The manifest file maps each key back to its tar path.
    """

    cache_dir: str = None
    manifest_path: str = None
    error_log_dir: str = None

    def _get_error_path(self, item: str) -> Path:
        return Path(self.error_log_dir) / item[:2] / f"{item}.log"

    def is_done(self, item: str) -> bool:
        from ludus_renderer.scene_cache import _cache_path, _resolve_cache_dir

        versioned_dir = _resolve_cache_dir(self.cache_dir)
        out_path = _cache_path(versioned_dir, item)
        if not out_path.exists():
            return False
        try:
            from ludus_renderer.scene_cache import load_scene_from_disk
            load_scene_from_disk(out_path)
            return True
        except Exception:
            out_path.unlink(missing_ok=True)
            return False

    def process(self, item: str):
        from ludus_renderer.clipgt import load_av2_scene
        from ludus_renderer.scene_cache import (
            _cache_path,
            _resolve_cache_dir,
            save_scene_to_disk,
        )

        tar_path = _lookup_manifest(self.manifest_path, item)

        versioned_dir = _resolve_cache_dir(self.cache_dir)
        out_path = _cache_path(versioned_dir, item)

        try:
            scene = load_av2_scene(tar_path, device="cpu")
            save_scene_to_disk(scene, out_path)
        except Exception:
            error_path = self._get_error_path(item)
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(
                f"scene_key: {item}\n"
                f"tar_path:  {tar_path}\n\n"
                f"{traceback.format_exc()}"
            )
            raise
