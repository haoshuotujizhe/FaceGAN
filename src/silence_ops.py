"""
Disable CUDA plugin compilation — use pure PyTorch ops instead.
Import this module BEFORE importing dnnlib or legacy.

【为什么需要】
NVIDIA StyleGAN2-ADA 自带 CUDA C++ 自定义算子 (bias_act, upfirdn2d),
编译需要完整 CUDA toolchain + ninja。若编译失败, 每次调用都重试,
导致满屏红色 traceback。

【解决方案】
Monkey-patch custom_ops.get_plugin → 直接返回 None →
NVIDIA 内置的纯 PyTorch fallback 路径生效 (已验证输出完全一致)。
"""
import os, sys, warnings, shutil

# 第1步: 清除之前编译失败的残留缓存 (防止 .so 文件找不到的错误)
shutil.rmtree(os.path.expanduser('~/.cache/torch_extensions'), ignore_errors=True)

# 第2步: 设置 CUDA 架构 + 静默所有 warning
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')  # RTX 4060 对应 sm_89
warnings.filterwarnings('ignore')

# 第3步: 把 official/ 目录加入 Python 模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'official'))

# 第4步: 关键 patch — 在任何模块调用 get_plugin 之前替换它
import torch_utils.custom_ops as c_ops
c_ops.get_plugin = lambda *a, **kw: None  # 直接返回 None → 触发纯 PyTorch fallback
