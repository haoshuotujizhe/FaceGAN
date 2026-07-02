"""
Generate figures for report/PPT. Output: figures/
Usage: python scripts/generate_figures.py
"""
import sys, os
sys.path.insert(0, 'official')
import torch, dnnlib, legacy
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = 'cuda'
os.makedirs('figures', exist_ok=True)

print('Loading model...')
with dnnlib.util.open_url('weights/ffhq.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()

gd = torch.load('weights/gender_dir_official.pt', map_location=device)
num_ws = G.mapping.num_ws

@torch.no_grad()
def gen(z=None, seed=None, alpha=0, psi=0.7):
    if seed is not None: torch.manual_seed(seed)
    if z is None: z = torch.randn(1, G.z_dim, device=device)
    ws = G.mapping(z, None, truncation_psi=psi)
    if alpha != 0: ws = ws + alpha * gd
    img = G.synthesis(ws, noise_mode='const')
    return ((img.squeeze(0).permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()

# ─── 1. 随机人脸 3×2 网格 ───
print('[1/5] 随机人脸网格')
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(gen(seed=i*10, psi=0.7))
    ax.axis('off'); ax.set_title(f'#{i+1}')
plt.tight_layout(); plt.savefig('figures/01_random_faces.png', dpi=200); plt.close()

# ─── 2. 性别渐变 1×9 ───
print('[2/5] 性别渐变')
torch.manual_seed(42)
z = torch.randn(1, G.z_dim, device=device)
ws = G.mapping(z, None, truncation_psi=0.7)
alphas = np.linspace(-4, 4, 9)
fig, axes = plt.subplots(1, 9, figsize=(22, 3))
for i, a in enumerate(alphas):
    img = G.synthesis(ws + a * gd, noise_mode='const')
    arr = ((img.squeeze(0).permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
    axes[i].imshow(arr); axes[i].axis('off')
    axes[i].set_title(f'{"M" if a<-0.5 else "F" if a>0.5 else "N"} a={a:+.0f}')
plt.tight_layout(); plt.savefig('figures/02_gender_walk.png', dpi=200); plt.close()

# ─── 3. 多身份对比 6×3 ───
print('[3/5] 多身份对比')
fig, axes = plt.subplots(6, 3, figsize=(10, 20))
for pid in range(6):
    torch.manual_seed(pid*17)
    ws = G.mapping(torch.randn(1, G.z_dim, device=device), None, truncation_psi=0.7)
    for j, (label, a) in enumerate([('M', -4), ('N', 0), ('F', +4)]):
        img = G.synthesis(ws + a * gd, noise_mode='const')
        arr = ((img.squeeze(0).permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
        axes[pid, j].imshow(arr); axes[pid, j].axis('off')
        if pid == 0: axes[pid, j].set_title(label, fontsize=14)
        if j == 0: axes[pid, j].set_ylabel(f'ID#{pid+1}', fontsize=12)
plt.tight_layout(); plt.savefig('figures/03_multi_identity.png', dpi=150); plt.close()

# ─── 4. 截断对比 1×6 ───
print('[4/5] 截断分析')
torch.manual_seed(999)
z = torch.randn(1, G.z_dim, device=device)
fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
for i, psi in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    ws = G.mapping(z, None, truncation_psi=psi)
    img = G.synthesis(ws, noise_mode='const')
    arr = ((img.squeeze(0).permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
    axes[i].imshow(arr); axes[i].axis('off')
    axes[i].set_title(f'ψ={psi}', fontsize=13)
plt.tight_layout(); plt.savefig('figures/04_truncation.png', dpi=200); plt.close()

# ─── 5. 系统架构图 (文本转图片) ───
print('[5/5] 系统架构图')
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
boxes = [
    (2, 5.0, 2, 0.6, 'z ~ N(0,I)\n(随机噪声 512D)', '#e3f2fd'),
    (2, 3.8, 2, 0.6, 'Mapping Network\n(8×Linear)', '#bbdefb'),
    (2, 2.6, 2, 0.6, 'W Space (512D)\n+ Gender Direction', '#90caf9'),
    (2, 1.4, 2, 0.6, 'Synthesis Network\n(18×ModConv)', '#64b5f6'),
    (2, 0.2, 2, 0.6, '1024×1024 RGB\n人脸图像', '#42a5f5'),
    (5, 2.6, 2.5, 0.6, 'Gender Ctrl\nw\' = w + α·d_gender\n(SVM + OpenCV)', '#ffccbc'),
    (7.5, 2.6, 2, 0.6, 'Gender Classifier\n(OpenCV CNN)', '#ffab91'),
]
arrows = [(3, 4.7, 2, 0.1), (3, 3.5, 2, 0.1), (3, 2.3, 2, 0.1), (3, 1.1, 2, 0.1)]
for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x-w/2, y-h/2), w, h, fill=True, facecolor=color,
                          edgecolor='#333', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, family='monospace')
for x1, y1, w, h in arrows:
    ax.annotate('', xy=(x1-0.3, y1-h/2), xytext=(x1-0.3, y1+h/2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.annotate('', xy=(3.5, 2.6), xytext=(4.5, 2.6),
            arrowprops=dict(arrowstyle='->', color='#e65100', lw=2.5))
ax.annotate('', xy=(6.2, 2.6), xytext=(7, 2.6),
            arrowprops=dict(arrowstyle='->', color='#e65100', lw=1.5, style='dashed'))
plt.tight_layout(); plt.savefig('figures/05_architecture.png', dpi=200); plt.close()

print(f'\nAll figures saved to figures/ ({len(os.listdir("figures"))} files)')
