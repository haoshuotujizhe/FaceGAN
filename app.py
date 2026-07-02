"""
🎭 FaceGAN 演示系统 — Gradio Web 界面
======================================
基于: NVIDIA StyleGAN2-ADA (官方) + InterFaceGAN 性别控制
用法: conda activate pytorch && pip install gradio && python app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'official'))

import torch, dnnlib, legacy
import numpy as np
from PIL import Image
import gradio as gr

# ─── 加载模型 ────────────────────────────────────────────

device = 'cuda'
print('📦 Loading StyleGAN2...')
with dnnlib.util.open_url('weights/ffhq.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
num_ws = G.mapping.num_ws
print(f'   {G.img_resolution}×{G.img_resolution} | W layers={num_ws}')

gd_path = 'weights/gender_dir_official.pt'
if os.path.exists(gd_path):
    gender_dir = torch.load(gd_path, map_location=device)
    print(f'   Gender direction: loaded ({gender_dir.shape})')
else:
    gender_dir = torch.randn(1, num_ws, G.z_dim, device=device)
    gender_dir = gender_dir / gender_dir.norm()
    print('   WARNING: using random direction')

# ─── 生成函数 ────────────────────────────────────────────

@torch.no_grad()
def generate(seed, gender_alpha, truncation_psi):
    """核心生成函数 — 先设种子, 再采样"""
    s = int(seed)
    if s >= 0:
        torch.manual_seed(s)
    z = torch.randn(1, G.z_dim, device=device)  # ← seed 之后才采样!

    ws = G.mapping(z, None, truncation_psi=float(truncation_psi))
    if abs(float(gender_alpha)) > 0.01:
        ws = ws + float(gender_alpha) * gender_dir

    img = G.synthesis(ws, noise_mode='const')
    arr = ((img.squeeze(0).permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
    return Image.fromarray(arr)


@torch.no_grad()
def generate_grid(seed, rows, cols, gender_alpha):
    """多张 → 拼成网格"""
    imgs = []
    base = int(seed) if seed >= 0 else 42
    for i in range(int(rows) * int(cols)):
        img = generate(base + i * 100, gender_alpha, 0.7)
        imgs.append(img)
    w, h = imgs[0].size
    grid = Image.new('RGB', (w * int(cols), h * int(rows)))
    for i, img in enumerate(imgs):
        r, c = divmod(i, int(cols))
        grid.paste(img, (c * w, r * h))
    return grid


@torch.no_grad()
def gender_walk(seed):
    """性别渐变: 7 帧拼一行"""
    s = int(seed)
    if s >= 0:
        torch.manual_seed(s)
    z = torch.randn(1, G.z_dim, device=device)
    ws = G.mapping(z, None, truncation_psi=0.7)

    frames = []
    for alpha in np.linspace(-4, 4, 7):
        img = G.synthesis(ws + alpha * gender_dir, noise_mode='const')
        arr = ((img.squeeze(0).permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
        frames.append(Image.fromarray(arr))
    w, h = frames[0].size
    gap = 10
    strip = Image.new('RGB', (w * 7 + gap * 6, h))
    for i, f in enumerate(frames):
        strip.paste(f, (i * (w + gap), 0))
    return strip


@torch.no_grad()
def trunc_grid(seed_val):
    """截断对比: ψ=0~1.0, 6 帧拼一行"""
    s = int(seed_val)
    if s >= 0:
        torch.manual_seed(s)
    z = torch.randn(1, G.z_dim, device=device)
    frames = []
    for psi in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        ws = G.mapping(z, None, truncation_psi=psi)
        img = G.synthesis(ws, noise_mode='const')
        arr = ((img.squeeze(0).permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
        frames.append(Image.fromarray(arr))
    w, h = frames[0].size
    gap = 8
    strip = Image.new('RGB', (w * 6 + gap * 5, h + 24))
    for i, f in enumerate(frames):
        strip.paste(f, (i * (w + gap), 24))
    return strip


def export_faces(seed, count):
    """批量导出"""
    os.makedirs('outputs', exist_ok=True)
    base = int(seed) if seed >= 0 else 0
    paths = []
    for i in range(int(count)):
        img = generate(base + i * 37, 0, 0.7)
        p = f'outputs/face_{i:03d}.png'
        img.save(p)
        paths.append(p)
    return f'✅ {count} 张 → outputs/', paths[0] if paths else None


# ─── Gradio UI ────────────────────────────────────────────

THEME = gr.themes.Soft(primary_hue="violet")
DEFAULT_IMG = generate(42, 0, 0.7)  # 初始默认图

with gr.Blocks(title="FaceGAN - 人脸生成演示系统", theme=THEME) as demo:
    gr.HTML("""
    <div style="text-align:center; margin-bottom:0.5em">
      <h1>🎭 基于 GAN 的人脸生成系统</h1>
      <p style="font-size:15px; color:#666">
        StyleGAN2-ADA (NVIDIA) + InterFaceGAN 潜空间操控 &nbsp;|&nbsp; 1024×1024 &nbsp;|&nbsp; FFHQ 预训练
      </p>
    </div>
    """)

    with gr.Tabs():
        # ═══════════════ Tab 1: 交互式控制 ═══════════════
        with gr.Tab("🎛️ 交互式控制"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 控制面板")
                    seed = gr.Slider(-1, 9999, value=42, step=1,
                                     label="🎲 种子 (-1=随机)")
                    gender = gr.Slider(-5, 5, value=0, step=0.2,
                                       label="👫 性别 (← 男 &nbsp;&nbsp; 女 →)")
                    trunc = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                      label="📐 截断 ψ (质量 ↔ 多样性)")

                    with gr.Row():
                        btn_update = gr.Button("🎨 更新生成", variant="primary", size="lg")
                        btn_random = gr.Button("🎲 随机", size="sm")

                    gr.Markdown("---")
                    gr.Markdown("**快捷预设:**")
                    with gr.Row():
                        btn_neutral = gr.Button("😐 中性", size="sm")
                        btn_male = gr.Button("👨 男性", size="sm")
                        btn_female = gr.Button("👩 女性", size="sm")

                with gr.Column(scale=1):
                    gr.Markdown("### 🖼️ 生成结果")
                    output_single = gr.Image(value=DEFAULT_IMG, label="")

            # ── 事件绑定 ──
            # 主按钮: 用当前 slider 值生成
            btn_update.click(generate, [seed, gender, trunc], output_single)
            btn_random.click(
                lambda: (generate(-1, 0, 0.7), -1, 0, 0.7),
                None, [output_single, seed, gender, trunc]
            )
            btn_neutral.click(
                lambda: (generate(42, 0, 0.7), 42, 0, 0.7),
                None, [output_single, seed, gender, trunc]
            )
            btn_male.click(
                lambda: (generate(-1, -4, 0.7), -1, -4, 0.7),
                None, [output_single, seed, gender, trunc]
            )
            btn_female.click(
                lambda: (generate(-1, 4, 0.7), -1, 4, 0.7),
                None, [output_single, seed, gender, trunc]
            )

        # ═══════════════ Tab 2: 性别渐变 ═══════════════
        with gr.Tab("🎞️ 性别渐变"):
            gr.Markdown("### 同一身份: 从男性到女性的连续 Morphing")
            with gr.Row():
                walk_seed = gr.Slider(-1, 9999, value=42, step=1, label="🎲 种子")
                walk_btn = gr.Button("▶️ 生成渐变", variant="primary")
            walk_output = gr.Image(label="👈 男 ——————————— 女 👉")
            walk_btn.click(gender_walk, walk_seed, walk_output)

        # ═══════════════ Tab 3: 批量画廊 ═══════════════
        with gr.Tab("🖼️ 批量画廊"):
            gr.Markdown("### 批量随机生成 → 拼成网格")
            with gr.Row():
                grid_seed = gr.Slider(-1, 9999, value=0, step=1, label="种子")
                grid_alpha = gr.Slider(-5, 5, value=0, step=0.2, label="性别")
                grid_rows = gr.Slider(1, 4, value=2, step=1, label="行数")
                grid_cols = gr.Slider(2, 5, value=3, step=1, label="列数")
            grid_btn = gr.Button("🎲 生成画廊", variant="primary")
            grid_output = gr.Image(label="")
            grid_btn.click(generate_grid,
                           [grid_seed, grid_rows, grid_cols, grid_alpha],
                           grid_output)

        # ═══════════════ Tab 4: 批量导出 ═══════════════
        with gr.Tab("💾 批量导出"):
            gr.Markdown("### 批量生成 → 保存到 `outputs/` 目录")
            with gr.Row():
                exp_seed = gr.Slider(-1, 9999, value=0, step=1, label="种子")
                exp_count = gr.Slider(4, 50, value=12, step=1, label="数量")
            exp_btn = gr.Button("📥 导出到本地", variant="primary")
            exp_msg = gr.Textbox(label="状态")
            exp_preview = gr.Image(label="预览 (第 1 张)")
            exp_btn.click(export_faces, [exp_seed, exp_count],
                          [exp_msg, exp_preview])

        # ═══════════════ Tab 5: 截断分析 ═══════════════
        with gr.Tab("📐 截断分析"):
            gr.Markdown("### 截断 ψ 对生成的影响")
            gr.Markdown("ψ=0 → 平均脸(质量高) &nbsp;|&nbsp; ψ=1 → 完全随机(多样性高)")
            with gr.Row():
                trunc_seed = gr.Slider(-1, 9999, value=42, step=1, label="种子")
                trunc_btn = gr.Button("📊 生成对比", variant="primary")
            trunc_output = gr.Image(label="ψ: 0.0 → 1.0")
            trunc_btn.click(trunc_grid, trunc_seed, trunc_output)

    gr.HTML("""
    <hr>
    <p style="text-align:center; color:#aaa; font-size:11px">
      FaceGAN | StyleGAN2-ADA (NVIDIA) + InterFaceGAN | FFHQ Pre-trained | PyTorch + Gradio
    </p>
    """)

if __name__ == '__main__':
    demo.launch(share=False, server_name='0.0.0.0', server_port=7860)
