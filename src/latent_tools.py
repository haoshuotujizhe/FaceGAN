"""
潜空间性别方向查找 (InterFaceGAN 方法)
====================================

原理 (Shen et al., CVPR 2020):
  StyleGAN2 的 W 空间中, 性别属性对应一个线性方向.
  w' = w + alpha * n_gender

主力实现在 scripts/compute_gender.py (CLIP + 10000样本 + SVM).
本文件保留核心 SVM 方向查找函数, 接收任意分类器作为参数.
"""

import torch
import numpy as np
from sklearn import svm
from tqdm import tqdm


def find_gender_direction(generator, classify_fn, num_samples=500, device='cuda'):
    """
    在 W 空间中查找性别方向.

    参数:
        generator:   StyleGAN2 Generator
        classify_fn: 分类函数, 输入 [B,3,H,W] tensor (值域[-1,1]), 返回 [B] bool (True=女性)
        num_samples: 采样数量 (推荐 10000)
        device:      'cuda'

    返回:
        direction: [1, 512] L2归一化性别方向向量
        info:      {'accuracy': ..., 'n_female': ..., 'n_male': ...}
    """
    generator.eval()
    generator.to(device)

    batch_size = 16
    all_w, all_labels = [], []

    print(f"[GenderDir] 采样 {num_samples} 个 W 编码...")
    with torch.no_grad():
        for _ in tqdm(range(num_samples // batch_size), desc="Sampling"):
            bs = batch_size
            z = torch.randn(bs, generator.style_dim, device=device)
            w = generator.style(z)                        # z → W
            all_w.append(w.cpu())

            w_plus = w.unsqueeze(1).repeat(1, generator.n_latent, 1)
            images, _ = generator([w_plus], input_is_latent=True)  # 生成人脸
            labels = classify_fn(images)                  # 分类器标注
            all_labels.append(labels.cpu())

    X = torch.cat(all_w).numpy()       # [N, 512] W编码
    y = torch.cat(all_labels).numpy()  # [N] 性别标签 (bool→int)

    n_f = int(y.sum())
    n_m = len(y) - n_f
    print(f"[GenderDir] ♀={n_f}, ♂={n_m}")

    # 线性 SVM → 决策超平面法向量 = 性别方向
    clf = svm.LinearSVC(C=1.0, max_iter=5000, dual=True, class_weight='balanced')
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"[GenderDir] SVM 准确率: {acc:.1%}")

    direction = torch.tensor(clf.coef_[0], dtype=torch.float32)
    direction = direction / (direction.norm() + 1e-8)
    direction = direction.unsqueeze(0).to(device)  # [1, 512]

    return direction, {'accuracy': acc, 'n_female': n_f, 'n_male': n_m}
