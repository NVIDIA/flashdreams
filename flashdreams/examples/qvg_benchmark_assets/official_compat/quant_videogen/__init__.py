"""Namespace shim so selected official QVG quant modules can be overridden."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
