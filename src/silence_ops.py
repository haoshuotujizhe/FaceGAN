"""
完全静默 CUDA 插件编译 — 从根源阻止, 纯 PyTorch 运行
必须在任何 official 模块 import 之前导入
"""
import os, sys, warnings, shutil

# 0. 清除坏缓存
shutil.rmtree(os.path.expanduser('~/.cache/torch_extensions'), ignore_errors=True)

# 1. 环境 & 静默
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')
warnings.filterwarnings('ignore')

# 2. 添加 official/ 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'official'))

# 3. 关键: 在任何人调用 get_plugin 之前, 替换它
import torch_utils.custom_ops as c_ops
c_ops.get_plugin = lambda *a, **kw: None
print('[silence_ops] CUDA 插件已禁用, 纯 PyTorch 运行')
