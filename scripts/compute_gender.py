"""
计算性别方向 v3 — CLIP 零样本分类 + SVM + 进度日志 + 断点保存
用法: pip install open-clip-torch && python scripts/compute_gender.py
"""
import sys, os, time
sys.path.insert(0, 'official'); sys.path.insert(0, '.')
import torch, dnnlib, legacy
import numpy as np
from tqdm import tqdm
from PIL import Image

device = 'cuda'
N = 10000
BATCH = 16
CHECKPOINT_EVERY = 1000  # 每 1000 张保存中间结果
CACHE_DIR = 'weights/cache_v3'

# ─── 1. 加载 StyleGAN2 ───
t0 = time.time()
print('='*60)
print('📦 [1/5] 加载 StyleGAN2...')
with dnnlib.util.open_url('weights/ffhq.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
num_ws = G.mapping.num_ws
print(f'     ✅ 完成 ({time.time()-t0:.1f}s) | {G.img_resolution}×{G.img_resolution}, W layers={num_ws}')

# ─── 2. 加载 CLIP ───
t1 = time.time()
print('📦 [2/5] 加载 CLIP ViT-B/32...')
import open_clip
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
clip_model = clip_model.to(device).eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
text_tokens = tokenizer(["a photo of a man", "a photo of a woman"]).to(device)
print(f'     ✅ 完成 ({time.time()-t1:.1f}s)')

# ─── 3. 批量采样 + CLIP 分类 ───
os.makedirs(CACHE_DIR, exist_ok=True)
total_batches = (N + BATCH - 1) // BATCH
print(f'🎲 [3/5] 采样 {N} 张 + CLIP 分类 (batch={BATCH}, 共{total_batches}批)')
print(f'     checkpoint 每{CHECKPOINT_EVERY}张保存到 {CACHE_DIR}/')
print()

w_list, labels = [], []
t_start = time.time()
female_running = 0

pbar = tqdm(total=N, desc='采样中', unit='张',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

for batch_idx, i in enumerate(range(0, N, BATCH)):
    bs = min(BATCH, N - i)
    z = torch.randn(bs, G.z_dim, device=device)
    ws = G.mapping(z, None)
    w0 = ws[:, 0, :].cpu()

    # 生成人脸
    imgs = G.synthesis(ws, noise_mode='const')

    # → PIL → CLIP preprocess
    pil_imgs = []
    for j in range(bs):
        arr = ((imgs[j].permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
        pil_imgs.append(preprocess(Image.fromarray(arr)))
    img_batch = torch.stack(pil_imgs).to(device)

    # CLIP 分类
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_batch)
        txt_feat = clip_model.encode_text(text_tokens)  # [2, 512]
        logits = img_feat @ txt_feat.T                   # [bs, 2]
        is_female = logits[:, 1] > logits[:, 0]

    w_list.append(w0)
    labels.extend(is_female.cpu().tolist())
    female_running += is_female.sum().item()

    # 进度更新
    current = i + bs
    elapsed = time.time() - t_start
    speed = current / elapsed
    eta = (N - current) / speed if speed > 0 else 0

    pbar.update(bs)
    pbar.set_postfix({
        '♀%': f'{female_running/current*100:.0f}%',
        '速度': f'{speed:.0f}张/s',
        'ETA': f'{eta/60:.0f}分{eta%60:.0f}秒' if eta < 3600 else f'{eta/3600:.1f}时'
    })

    # 定期保存 checkpoint
    if current % CHECKPOINT_EVERY == 0:
        ckpt = {
            'w': torch.cat(w_list).numpy(),
            'labels': np.array(labels, dtype=int),
            'n_done': current,
            'female_pct': female_running / current
        }
        torch.save(ckpt, f'{CACHE_DIR}/ckpt_{current:05d}.pt')
        tqdm.write(f'     💾 checkpoint @ {current}/{N} (♀={female_running/current*100:.1f}%)')

pbar.close()

X = torch.cat(w_list).numpy()
y = np.array(labels, dtype=int)
n_f = int(y.sum()); n_m = N - n_f

elapsed_total = time.time() - t_start
print(f'\n📊 CLIP 分类完成: ♀={n_f} ({n_f/N:.0%})  ♂={n_m} ({n_m/N:.0%})')
print(f'    耗时 {elapsed_total/60:.1f} 分钟 ({elapsed_total/N:.2f}秒/张)')

# ─── 4. SVM 找方向 ───
t3 = time.time()
print(f'\n🔬 [4/5] SVM 训练 (LinearSVC, {N}样本 × 512维)...')
from sklearn.svm import LinearSVC
svm = LinearSVC(C=1.0, max_iter=5000, dual=True, class_weight='balanced')
svm.fit(X, y)
acc = svm.score(X, y)
print(f'     ✅ SVM 准确率: {acc:.2%}  ({time.time()-t3:.1f}s)')

dir_raw = torch.tensor(svm.coef_[0], dtype=torch.float32).to(device).unsqueeze(0)
dir_raw = dir_raw / dir_raw.norm()
print(f'     方向向量 norm={dir_raw.norm():.4f}, 有效维度={(dir_raw.abs()>0.01).sum().item()}/512')

# ─── 5. 方向验证 ───
t4 = time.time()
print(f'\n🔍 [5/5] CLIP 验证方向符号 + 保存...')
with torch.no_grad():
    z_test = torch.randn(8, G.z_dim, device=device)
    ws_test = G.mapping(z_test, None)
    gdir = dir_raw.unsqueeze(1).repeat(1, num_ws, 1)

    imgs_pos = G.synthesis(ws_test + 4.0 * gdir, noise_mode='const')
    imgs_neg = G.synthesis(ws_test - 4.0 * gdir, noise_mode='const')

    def clip_verify(imgs):
        pil_imgs = []
        for j in range(imgs.size(0)):
            arr = ((imgs[j].permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
            pil_imgs.append(preprocess(Image.fromarray(arr)))
        batch = torch.stack(pil_imgs).to(device)
        feat = clip_model.encode_image(batch)
        txt = clip_model.encode_text(text_tokens)  # [2, 512]
        lg = feat @ txt.T                             # [8, 2]
        return lg[:, 1] > lg[:, 0]

    pos_f = clip_verify(imgs_pos).sum().item()
    neg_m = (~clip_verify(imgs_neg)).sum().item()
    print(f'     α>0 → 女性: {pos_f}/8  |  α<0 → 男性: {neg_m}/8')

if pos_f < 6:
    print('     ⚠️ 方向翻转!')
    dir_raw = -dir_raw

# ─── 6. 保存最终结果 ───
strength_factor = 2.5
dir_w = dir_raw * strength_factor
gender_dir = dir_w.unsqueeze(1).repeat(1, num_ws, 1)

os.makedirs('weights', exist_ok=True)
torch.save(gender_dir, 'weights/gender_dir_official.pt')
print(f'     ✅ 已保存 weights/gender_dir_official.pt')
print(f'        CLIP+SVM | N={N} | SVM acc={acc:.1%} | x{strength_factor:.1f}')

# ─── 7. 测试图 ───
print(f'\n🎨 生成验证图...')
import matplotlib.pyplot as plt
torch.manual_seed(42)
z = torch.randn(1, G.z_dim, device=device)
ws = G.mapping(z, None)

fig, axes = plt.subplots(1, 9, figsize=(24, 3))
for i, a in enumerate(np.linspace(-4, 4, 9)):
    img = G.synthesis(ws + a * gender_dir, noise_mode='const')
    arr = ((img.squeeze(0).permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
    axes[i].imshow(arr); axes[i].axis('off')
    emoji = '👨' if a < -0.5 else '👩' if a > 0.5 else '😐'
    axes[i].set_title(f'{emoji} α={a:+.0f}', fontsize=10)
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/gender_test_v3.png', bbox_inches='tight', pad_inches=0, dpi=200)
print('✅ figures/gender_test_v3.png')

# ─── 清理 checkpoint ───
if os.path.exists(CACHE_DIR):
    import shutil
    shutil.rmtree(CACHE_DIR)
    print(f'🧹 已清理临时 checkpoint {CACHE_DIR}')

print()
print('='*60)
print(f'🎉 全部完成! 总耗时 { (time.time()-t0)/60:.1f} 分钟')
print(f'   N={N} | SVM acc={acc:.1%} | 方向验证 {pos_f}/{neg_m} ✓')
print('='*60)
