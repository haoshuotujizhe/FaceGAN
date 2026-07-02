"""下载 NVIDIA 官方 StyleGAN2-ADA FFHQ 权重 (~300MB)"""
import os, requests, tqdm, sys

url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
out_path = 'weights/ffhq.pkl'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

if os.path.exists(out_path):
    print(f'✅ 已存在: {out_path} ({os.path.getsize(out_path)/1024**2:.0f}MB)')
    sys.exit(0)

print(f'📥 下载 FFHQ 预训练权重 (~300MB)...')
r = requests.get(url, stream=True)
total = int(r.headers.get('content-length', 0))
with open(out_path, 'wb') as f:
    for chunk in tqdm.tqdm(r.iter_content(chunk_size=8192),
                           total=total//8192, unit='KB'):
        f.write(chunk)
print(f'✅ 下载完成: {out_path} ({os.path.getsize(out_path)/1024**2:.0f}MB)')
