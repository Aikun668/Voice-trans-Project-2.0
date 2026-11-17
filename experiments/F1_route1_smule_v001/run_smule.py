# 运行器：以隔离模式(-I)运行官方 main.py，仅显式补上官方仓库路径。
# 原因：-I 不把脚本目录加入 sys.path，官方 main.py 的
# `from spectral_ops import STFT, iSTFT` 会失败。不改动官方代码。
import sys
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "smule-renaissance"))
runpy.run_path(str(ROOT / "smule-renaissance" / "main.py"), run_name="__main__")
