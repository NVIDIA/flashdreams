"""Namespace shim for official QVG real-quant module overrides."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
