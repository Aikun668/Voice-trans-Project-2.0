"""导入 vendored nara_wpe 0.0.11（MIT, https://github.com/fgnt/nara_wpe）。
wpe.py 模块级用 click 装饰其命令行入口；本环境无 click，且算法本身不依赖它，
故注入只返回原函数的空装饰器，不安装额外依赖、不改动 vendored 源码。"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
if "click" not in sys.modules:
    _stub = types.ModuleType("click")
    _deco = lambda *a, **k: (lambda f: f)  # noqa: E731
    _stub.command = _stub.option = _stub.argument = _deco
    _stub.Path = lambda *a, **k: None
    sys.modules["click"] = _stub

from nara_wpe.utils import istft, stft  # noqa: E402
from nara_wpe.wpe import wpe  # noqa: E402

__all__ = ["wpe", "stft", "istft"]
