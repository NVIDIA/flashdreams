from __future__ import annotations

from pathlib import Path

import pytest

from wan22.apps.ti2v import create_app

pytestmark = pytest.mark.ci_cpu


class _PipelineConfig:
    def setup(self) -> object:
        raise AssertionError("pipeline setup must be deferred to the model loop")


def test_ti2v_registers_native_model_and_slangpy_ui_loops(tmp_path: Path) -> None:
    image = tmp_path / "first.png"
    image.touch()
    app = create_app(
        pipeline_config=_PipelineConfig(),
        ui_renderer_factory=lambda width, height: object(),
    )
    app.init(["--image-path", str(image), "--prompt", "A moving camera"])
    session = app.create_session(app.session_desc())
    session.init()
    ui_loop, model_loop = session._take_loops()
    assert type(ui_loop).__name__ == "T2VSlangPyUILoop"
    assert type(model_loop).__name__ == "T2VModelLoop"


def test_ti2v_rejects_more_than_one_block(tmp_path: Path) -> None:
    image = tmp_path / "first.png"
    image.touch()
    app = create_app(pipeline_config=_PipelineConfig())
    with pytest.raises(ValueError, match="exactly one block"):
        app.init(["--image-path", str(image), "--total-blocks", "2"])
