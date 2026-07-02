"""
FaceGAN — StyleGAN2 Face Generation with Gender Control
Usage: conda activate pytorch && python app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'official'))

import torch, dnnlib, legacy
import numpy as np
from PIL import Image
import gradio as gr

# -- Load model -------------------------------------------------

device = 'cuda'
print('Loading StyleGAN2-ADA...')
with dnnlib.util.open_url('weights/ffhq.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
num_ws = G.mapping.num_ws
print(f'  {G.img_resolution}x{G.img_resolution}, W layers={num_ws}')

gd_path = 'weights/gender_dir_official.pt'
if os.path.exists(gd_path):
    gender_dir = torch.load(gd_path, map_location=device)
    print(f'  Gender direction loaded: {gender_dir.shape}')
else:
    gender_dir = torch.randn(1, num_ws, G.z_dim, device=device)
    gender_dir = gender_dir / gender_dir.norm()
    print('  Warning: using random gender direction')

# -- Generation functions ---------------------------------------

@torch.no_grad()
def generate(seed, gender_alpha, truncation_psi):
    s = int(seed)
    if s >= 0:
        torch.manual_seed(s)
    z = torch.randn(1, G.z_dim, device=device)

    ws = G.mapping(z, None, truncation_psi=float(truncation_psi))
    if abs(float(gender_alpha)) > 0.01:
        ws = ws + float(gender_alpha) * gender_dir

    img = G.synthesis(ws, noise_mode='const')
    arr = ((img.squeeze(0).permute(1,2,0).cpu()+1)*127.5).clamp(0,255).to(torch.uint8).numpy()
    return Image.fromarray(arr)


@torch.no_grad()
def generate_grid(seed, rows, cols, gender_alpha):
    imgs = []
    base = int(seed) if seed >= 0 else 42
    for i in range(int(rows) * int(cols)):
        imgs.append(generate(base + i * 100, gender_alpha, 0.7))
    w, h = imgs[0].size
    grid = Image.new('RGB', (w * int(cols), h * int(rows)))
    for i, img in enumerate(imgs):
        r, c = divmod(i, int(cols))
        grid.paste(img, (c * w, r * h))
    return grid


@torch.no_grad()
def gender_walk(seed):
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
    os.makedirs('outputs', exist_ok=True)
    base = int(seed) if seed >= 0 else 0
    paths = []
    for i in range(int(count)):
        img = generate(base + i * 37, 0, 0.7)
        p = f'outputs/face_{i:03d}.png'
        img.save(p)
        paths.append(p)
    return f'Saved {count} images to outputs/', paths[0] if paths else None


# -- Gradio UI -------------------------------------------------

THEME = gr.themes.Soft(primary_hue="violet")
DEFAULT_IMG = generate(42, 0, 0.7)

with gr.Blocks(title="FaceGAN - Face Generation System", theme=THEME) as demo:
    gr.HTML("""
    <div style="text-align:center; margin-bottom:0.5em">
      <h1>基于 GAN 的人脸生成系统</h1>
      <p style="font-size:15px; color:#666">
        StyleGAN2-ADA (NVIDIA) + InterFaceGAN 潜空间操控 | 1024x1024 | FFHQ 预训练
      </p>
    </div>
    """)

    with gr.Tabs():
        # Tab 1: Interactive Control
        with gr.Tab("Interactive"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Controls")
                    seed = gr.Slider(-1, 9999, value=42, step=1,
                                     label="Seed (-1 = random)")
                    gender = gr.Slider(-5, 5, value=0, step=0.2,
                                       label="Gender (left=male, right=female)")
                    trunc = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                      label="Truncation psi (quality vs. diversity)")

                    with gr.Row():
                        btn_update = gr.Button("Generate", variant="primary", size="lg")
                        btn_random = gr.Button("Random", size="sm")

                    gr.Markdown("---")
                    gr.Markdown("**Presets:**")
                    with gr.Row():
                        btn_neutral = gr.Button("Neutral", size="sm")
                        btn_male = gr.Button("Male", size="sm")
                        btn_female = gr.Button("Female", size="sm")

                with gr.Column(scale=1):
                    gr.Markdown("### Result")
                    output_single = gr.Image(value=DEFAULT_IMG, label="")

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

        # Tab 2: Gender Walk
        with gr.Tab("Gender Walk"):
            gr.Markdown("### Continuous morphing from male to female")
            with gr.Row():
                walk_seed = gr.Slider(-1, 9999, value=42, step=1, label="Seed")
                walk_btn = gr.Button("Generate Walk", variant="primary")
            walk_output = gr.Image(label="Male  ----------  Female")
            walk_btn.click(gender_walk, walk_seed, walk_output)

        # Tab 3: Gallery
        with gr.Tab("Gallery"):
            gr.Markdown("### Random face gallery")
            with gr.Row():
                grid_seed = gr.Slider(-1, 9999, value=0, step=1, label="Seed")
                grid_alpha = gr.Slider(-5, 5, value=0, step=0.2, label="Gender")
                grid_rows = gr.Slider(1, 4, value=2, step=1, label="Rows")
                grid_cols = gr.Slider(2, 5, value=3, step=1, label="Cols")
            grid_btn = gr.Button("Generate Gallery", variant="primary")
            grid_output = gr.Image(label="")
            grid_btn.click(generate_grid,
                           [grid_seed, grid_rows, grid_cols, grid_alpha],
                           grid_output)

        # Tab 4: Export
        with gr.Tab("Export"):
            gr.Markdown("### Batch export to outputs/")
            with gr.Row():
                exp_seed = gr.Slider(-1, 9999, value=0, step=1, label="Seed")
                exp_count = gr.Slider(4, 50, value=12, step=1, label="Count")
            exp_btn = gr.Button("Export", variant="primary")
            exp_msg = gr.Textbox(label="Status")
            exp_preview = gr.Image(label="Preview (first image)")
            exp_btn.click(export_faces, [exp_seed, exp_count],
                          [exp_msg, exp_preview])

        # Tab 5: Truncation Analysis
        with gr.Tab("Truncation"):
            gr.Markdown("### Effect of truncation psi")
            gr.Markdown("psi=0: average face (high quality) | psi=1: fully random (high diversity)")
            with gr.Row():
                trunc_seed = gr.Slider(-1, 9999, value=42, step=1, label="Seed")
                trunc_btn = gr.Button("Generate Comparison", variant="primary")
            trunc_output = gr.Image(label="psi: 0.0 -> 1.0")
            trunc_btn.click(trunc_grid, trunc_seed, trunc_output)

    gr.HTML("""
    <hr>
    <p style="text-align:center; color:#aaa; font-size:11px">
      FaceGAN | StyleGAN2-ADA (NVIDIA) + InterFaceGAN | FFHQ | PyTorch + Gradio
    </p>
    """)

if __name__ == '__main__':
    demo.launch(share=False, server_name='0.0.0.0', server_port=7860)
