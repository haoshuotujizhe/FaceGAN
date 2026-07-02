"""Disable CUDA plugin compilation — use pure PyTorch ops instead.
Import this module BEFORE importing dnnlib or legacy.
"""
import os, sys, warnings, shutil

# Clear stale build cache
shutil.rmtree(os.path.expanduser('~/.cache/torch_extensions'), ignore_errors=True)

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')
warnings.filterwarnings('ignore')

# Add official/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'official'))

# Patch get_plugin before any module calls it
import torch_utils.custom_ops as c_ops
c_ops.get_plugin = lambda *a, **kw: None
