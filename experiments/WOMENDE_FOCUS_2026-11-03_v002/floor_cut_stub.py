"""占位，不是 v002 的原始实现。

本仓库快照没有 experiments/F1_demo_chain_2026-11-02_v002/run.py，
而 v003 的 chain_dry 在导入时要取 floor_cut / chain_v3 / target_spectrum。
这里的 floor_cut 原样返回已处理音频，不追加句间降噪。
chain_v3、target_spectrum 本轮不用，调用即报错，避免被误当成原实现。
"""


def floor_cut(orig, proc, sr):
    return proc


def chain_v3(*_args, **_kwargs):
    raise RuntimeError("v002 chain_v3 不在本快照中，本轮没有调用它")


def target_spectrum(*_args, **_kwargs):
    raise RuntimeError("v002 target_spectrum 不在本快照中，本轮没有调用它")
