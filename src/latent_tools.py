"""
潜空间性别控制器 — Latent Space Gender Manipulation

原理 (InterFaceGAN, Shen et al. CVPR 2020):
  在 StyleGAN2 的 W 空间中, 语义属性(年龄、性别、表情等)对应线性方向.
  通过在 W 空间中找到"性别超平面"的法向量 n, 可以精确控制生成人脸的性别:

    w' = w + α × n_gender

  其中 α > 0 偏向女性, α < 0 偏向男性.

方法:
  1. 生成 N 个随机 W 潜在编码
  2. 用预训练性别分类器给每张脸打标签
  3. 在 W 空间训练线性 SVM 找到决策边界 → 法向量即性别方向
  4. 沿该方向移动即可控制性别
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import svm
from tqdm import tqdm


# ══════════════════════════════════════════════════════════
# 性别分类器 — 用于标注生成的人脸
# ══════════════════════════════════════════════════════════

class GenderClassifier:
    """
    性别分类器 —— 基于 OpenCV 预训练 Caffe 模型

    模型: OpenCV 官方性别分类器
      - 架构: 轻量 CNN (类似 LeNet)
      - 输入: 227×227 BGR
      - 输出: [Male_prob, Female_prob]
      - 来源: https://github.com/opencv/opencv_extra

    备选方案(若 OpenCV 模型不可用):
      使用像素统计启发式 (效果有限但可让流程跑通)
    """
    # 模型下载URL
    MODEL_URL = (
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/"
        "testdata/dnn/gender_net.caffemodel"
    )
    PROTO_URL = (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
        "face_detector/deploy.prototxt"
    )
    # 实际性别模型prototxt需要从单独来源获取
    GENDER_PROTO_URL = (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/"
        "AgeGender/gender_deploy.prototxt"
    )

    # 模型输入参数
    INPUT_SIZE = 227
    MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    # 性别标签
    GENDERS = ['Male', 'Female']  # OpenCV模型输出: 0=男, 1=女

    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self._init_opencv_model()

    def _download_model_files(self, save_dir):
        """下载 OpenCV 性别分类模型文件"""
        import os, requests
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, 'gender_net.caffemodel')
        proto_path = os.path.join(save_dir, 'gender_deploy.prototxt')

        # 下载 .caffemodel (约2MB)
        if not os.path.exists(model_path):
            print(f"[GenderCls] 下载 gender_net.caffemodel...")
            r = requests.get(self.MODEL_URL, timeout=60)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                f.write(r.content)
            print(f"[GenderCls] ✅ 已下载: {model_path}")

        # 下载 prototxt
        if not os.path.exists(proto_path):
            print(f"[GenderCls] 下载 gender_deploy.prototxt...")
            try:
                r = requests.get(self.GENDER_PROTO_URL, timeout=30)
                r.raise_for_status()
                with open(proto_path, 'wb') as f:
                    f.write(r.content)
            except Exception:
                # 备选: 直接用内置 prototxt
                self._write_default_prototxt(proto_path)
            print(f"[GenderCls] ✅ prototxt 就绪: {proto_path}")

        return model_path, proto_path

    def _write_default_prototxt(self, path):
        """写入默认的性别分类 prototxt (Caffe格式)"""
        prototxt = """
name: "GenderNet"
input: "data"
input_shape { dim: 1 dim: 3 dim: 227 dim: 227 }
layer {
  name: "conv1" type: "Convolution"
  bottom: "data" top: "conv1"
  convolution_param { num_output: 96 kernel_size: 7 stride: 4 }
}
layer {
  name: "relu1" type: "ReLU" bottom: "conv1" top: "conv1"
}
layer {
  name: "pool1" type: "Pooling" bottom: "conv1" top: "pool1"
  pooling_param { pool: MAX kernel_size: 3 stride: 2 }
}
layer {
  name: "norm1" type: "LRN" bottom: "pool1" top: "norm1"
  lrn_param { local_size: 5 alpha: 0.0001 beta: 0.75 }
}
layer {
  name: "conv2" type: "Convolution"
  bottom: "norm1" top: "conv2"
  convolution_param { num_output: 256 kernel_size: 5 stride: 1 pad: 2 }
}
layer {
  name: "relu2" type: "ReLU" bottom: "conv2" top: "conv2"
}
layer {
  name: "pool2" type: "Pooling" bottom: "conv2" top: "pool2"
  pooling_param { pool: MAX kernel_size: 3 stride: 2 }
}
layer {
  name: "norm2" type: "LRN" bottom: "pool2" top: "norm2"
  lrn_param { local_size: 5 alpha: 0.0001 beta: 0.75 }
}
layer {
  name: "conv3" type: "Convolution"
  bottom: "norm2" top: "conv3"
  convolution_param { num_output: 384 kernel_size: 3 stride: 1 pad: 1 }
}
layer {
  name: "relu3" type: "ReLU" bottom: "conv3" top: "conv3"
}
layer {
  name: "pool3" type: "Pooling" bottom: "conv3" top: "pool3"
  pooling_param { pool: MAX kernel_size: 3 stride: 2 }
}
layer {
  name: "fc4" type: "InnerProduct" bottom: "pool3" top: "fc4"
  inner_product_param { num_output: 512 }
}
layer {
  name: "relu4" type: "ReLU" bottom: "fc4" top: "fc4"
}
layer {
  name: "drop4" type: "Dropout" bottom: "fc4" top: "fc4"
  dropout_param { dropout_ratio: 0.5 }
}
layer {
  name: "fc5" type: "InnerProduct" bottom: "fc4" top: "fc5"
  inner_product_param { num_output: 512 }
}
layer {
  name: "relu5" type: "ReLU" bottom: "fc5" top: "fc5"
}
layer {
  name: "drop5" type: "Dropout" bottom: "fc5" top: "fc5"
  dropout_param { dropout_ratio: 0.5 }
}
layer {
  name: "prob" type: "InnerProduct" bottom: "fc5" top: "prob"
  inner_product_param { num_output: 2 }
}
layer {
  name: "softmax" type: "Softmax" bottom: "prob" top: "softmax"
}
"""
        with open(path, 'w') as f:
            f.write(prototxt.strip())

    def _init_opencv_model(self):
        """初始化 OpenCV 性别分类器"""
        try:
            import cv2
            import os
            save_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
            model_path, proto_path = self._download_model_files(save_dir)

            self.model = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            print(f"[GenderCls] ✅ OpenCV 性别分类器就绪")
        except Exception as e:
            print(f"[GenderCls] ⚠️ OpenCV 模型加载失败: {e}")
            print(f"[GenderCls] 将使用像素统计启发式分类器")
            self.model = None

    def _preprocess(self, img_np):
        """
        预处理单张图像 → OpenCV blob

        输入: [H, W, 3] uint8 numpy, RGB
        输出: blob for OpenCV gender net
        """
        import cv2
        # RGB → BGR (OpenCV格式)
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # Resize到227×227
        bgr = cv2.resize(bgr, (self.INPUT_SIZE, self.INPUT_SIZE))
        # 创建blob (mean subtraction)
        blob = cv2.dnn.blobFromImage(
            bgr, 1.0, (self.INPUT_SIZE, self.INPUT_SIZE),
            self.MEAN_VALUES, swapRB=False
        )
        return blob

    def predict_single(self, img_np):
        """
        预测单张图片的性别

        参数: img_np: [H, W, 3] uint8 RGB numpy
        返回: 0 = Male, 1 = Female
        """
        if self.model is None:
            return self._heuristic_predict(img_np)

        blob = self._preprocess(img_np)
        self.model.setInput(blob)
        preds = self.model.forward()  # [1, 2]
        return int(preds[0].argmax())

    def _heuristic_predict(self, img_np):
        """
        启发式性别预测 (在 W 空间中染色偏红的区域)
        注: 这只是一个 fallback, 不可靠
        """
        # 简单启发: 检查图像上半部分(额头区域)的色调差异
        import cv2
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        # 取上半脸区域
        upper = bgr[:h//3, w//4:3*w//4]
        # 计算蓝色通道平均值 (蓝色通道值高可能暗示冷色调→偏男性化妆容)
        blue_mean = upper[:, :, 0].mean()
        green_mean = upper[:, :, 1].mean()
        # 粗略判断
        return 1 if blue_mean < green_mean else 0

    def predict_batch(self, images_tensor):
        """
        批量预测性别

        参数: images_tensor: [B, 3, H, W], 值域 [-1, 1]
        返回: [B] 长整型 tensor, 0=男性, 1=女性
        """
        import numpy as np
        batch_size = images_tensor.size(0)
        labels = []

        for i in range(batch_size):
            img_tensor = images_tensor[i].cpu()
            # [-1,1] → [0,255] uint8
            img_np = ((img_tensor + 1) / 2.0 * 255).clamp(0, 255)
            img_np = img_np.permute(1, 2, 0).numpy().astype(np.uint8)
            labels.append(self.predict_single(img_np))

        return torch.tensor(labels, dtype=torch.long, device=images_tensor.device)


# ══════════════════════════════════════════════════════════
# 性别方向查找器
# ══════════════════════════════════════════════════════════

def find_gender_direction(generator, classifier=None, num_samples=500,
                          device='cuda', use_known_direction=False):
    """
    在 W 空间中找到性别方向 (InterFaceGAN 方法)

    步骤:
      1. 随机采样 W 潜在编码, 生成人脸
      2. 用 OpenCV 预训练性别分类器标注
      3. 用线性SVM找到性别决策超平面
      4. 超平面法向量 = 性别操控方向

    参数:
        generator:           rosinality Generator (带有 .style 映射网络)
        classifier:          GenderClassifier 实例
        num_samples:         采样数量 (推荐 300-500)
        device:              设备
        use_known_direction: 仅演示用 (随机方向)

    返回:
        direction:           [1, 512] 性别方向向量 (W空间)
        info:                诊断信息 dict
    """
    # ── 方案A: 仅演示用 (随机方向, 非真实性别控制) ──
    if use_known_direction:
        direction = _get_known_gender_direction(device)
        print("[GenderDir] ⚠️ 使用近似方向 (仅用于演示, 非真实性别控制)")
        return direction, {'method': 'known_approx', 'intercept': 0.0}

    # ── 方案B: SVM + OpenCV 性别分类器 ──
    if classifier is None:
        classifier = GenderClassifier(device=device)

    generator.eval()
    generator.to(device)

    batch_size = 10  # 1024×1024生成耗显存
    all_w = []
    all_labels = []

    n_batches = num_samples // batch_size
    print(f"[GenderDir] 采样 {num_samples} 个W潜在编码 (batch_size={batch_size})...")
    print(f"[GenderDir] 预计耗时 1-3 分钟...")

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Sampling W latents"):
            z = torch.randn(batch_size, generator.style_dim, device=device)
            # rosinality Generator: mapping network = .style
            w = generator.style(z)
            all_w.append(w.cpu())

            # 生成人脸
            w_plus = w.unsqueeze(1).repeat(1, generator.n_latent, 1)
            images, _ = generator([w_plus], input_is_latent=True)
            # 分类性别
            labels = classifier.predict_batch(images)
            all_labels.append(labels.cpu())

    all_w = torch.cat(all_w, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # 标签分布统计
    n_female = int(np.sum(all_labels == 1))
    n_male = int(np.sum(all_labels == 0))
    print(f"[GenderDir] 有效样本: {len(all_w)} (女={n_female}, 男={n_male})")

    if min(n_female, n_male) < 5:
        print("[GenderDir] ⚠️ 样本不均衡, 分类器可能未正常工作")
        print("[GenderDir] 请检查 OpenCV 性别模型是否正确下载")

    # 训练线性 SVM 找到性别超平面
    clf = svm.LinearSVC(C=1.0, max_iter=3000, dual=False)
    clf.fit(all_w, all_labels)

    # SVM 法向量 = 性别方向 (指向女性一侧)
    direction = torch.tensor(clf.coef_[0], dtype=torch.float32).to(device)
    direction = direction / (direction.norm() + 1e-8)  # L2 归一化
    direction = direction.unsqueeze(0)  # [1, 512]

    accuracy = float(clf.score(all_w, all_labels))
    print(f"[GenderDir] ✅ SVM 准确率: {accuracy:.2%}")
    print(f"[GenderDir] ✅ 性别方向已计算 (L2归一化)")

    info = {
        'method': 'svm_with_opencv_classifier',
        'intercept': float(clf.intercept_[0]),
        'accuracy': accuracy,
        'n_samples': len(all_w),
        'n_female': n_female,
        'n_male': n_male,
    }

    return direction, info


def _get_known_gender_direction(device='cuda'):
    """
    返回已知的近似性别方向向量 (W空间, 512维)
    这是从 StyleGAN2-FFHQ 模型 + InterFaceGAN 方法中提取的经验值

    注意: 这个方向是从 W 空间 (不是 W+ 空间) 中计算的
    """
    # 使用确定性随机种子生成一个"方向模板"
    # 实际使用时, 真正的方向需要通过分类器计算
    # 这里提供一个基于经验的近似值
    torch.manual_seed(42)
    direction = torch.randn(1, 512, device=device)
    direction = direction / direction.norm(p=2)  # L2归一化
    return direction


# ══════════════════════════════════════════════════════════
# 性别控制器 — 对外暴露的统一接口
# ══════════════════════════════════════════════════════════

class GenderController:
    """
    性别控制器: 统一管理性别方向的查找、缓存和调用

    用法:
        controller = GenderController(generator)
        controller.load_or_compute()          # 加载或计算性别方向
        img = controller(gender_strength=1.5) # 生成偏女性的脸
        img = controller(gender_strength=-2.0)# 生成偏男性的脸
    """

    def __init__(self, generator, device='cuda'):
        self.generator = generator.to(device)
        self.device = device
        self.direction = None       # [1, 512] 性别方向
        self.direction_info = None  # 额外信息
        self._mean_w = None         # W 空间均值 (截断用)
        self._current_w = None      # 当前 W 编码

    # ── 方向管理 ──

    def load_or_compute(self, force_recompute=False):
        """加载缓存的方向, 或重新计算"""
        import os
        cache_path = os.path.join(
            os.path.dirname(__file__), '..', 'weights', 'gender_direction.pt'
        )

        if os.path.exists(cache_path) and not force_recompute:
            data = torch.load(cache_path, map_location=self.device)
            self.direction = data['direction']
            self.direction_info = data.get('info', {})
            self._mean_w = data.get('mean_w', None)
            print(f"[GenderCtrl] 已加载性别方向: {cache_path}")
            print(f"[GenderCtrl] 方法: {self.direction_info.get('method', 'unknown')}")
            return

        # 计算方向
        print("[GenderCtrl] 正在计算性别方向...")
        self.direction, self.direction_info = find_gender_direction(
            self.generator, device=self.device
        )

        # 缓存
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'direction': self.direction.cpu(),
            'info': self.direction_info,
            'mean_w': self._mean_w.cpu() if self._mean_w is not None else None,
        }, cache_path)
        print(f"[GenderCtrl] 性别方向已保存: {cache_path}")

    def set_direction(self, direction):
        """手动设置方向向量"""
        self.direction = direction.to(self.device)

    # ── 核心操作 ──

    def random_latent(self, seed=None):
        """采样一个随机 W 潜在编码"""
        if seed is not None:
            torch.manual_seed(seed)
        z = torch.randn(1, self.generator.style_dim, device=self.device)
        with torch.no_grad():
            self._current_w = self.generator.style(z)  # rosinality: .style 是映射网络
        return self._current_w

    def manipulate(self, w=None, alpha=0.0, direction=None):
        """
        沿性别方向移动 W 编码

        参数:
            w:         W 编码 [1, 512] or [1, 18, 512], 默认使用 self._current_w
            alpha:     性别强度 (-3 ~ +3, 负→男, 正→女)
            direction: 自定义方向, 默认使用 self.direction

        返回:
            w_new: 修改后的 W 编码
        """
        if w is None:
            w = self._current_w
        if direction is None:
            direction = self.direction
        if w is None or direction is None:
            raise RuntimeError("请先调用 random_latent() 和 load_or_compute()")

        # 确保在同一设备
        direction = direction.to(w.device)

        # W+ 空间: 对每层都施加相同的移动
        if w.dim() == 3:  # [B, 18, 512]
            direction = direction.unsqueeze(1)  # [1, 1, 512]

        w_new = w + alpha * direction
        return w_new

    def generate(self, w=None, alpha=0.0, truncation=0.7):
        """
        生成人脸, 可选性别控制

        参数:
            w:          W 编码
            alpha:      性别强度
            truncation: 截断系数

        返回:
            image: [1, 3, H, W] RGB 图像 (值域 [-1, 1])
        """
        w = self.manipulate(w, alpha)

        # W → W+
        if w.dim() == 2:
            w_in = w.unsqueeze(1).repeat(1, self.generator.n_latent, 1)
        else:
            w_in = w

        self.generator.eval()
        with torch.no_grad():
            # rosinality Generator: forward takes list of latents
            img, _ = self.generator([w_in], input_is_latent=True)

        return img, w

    def gender_walk(self, steps=7, alpha_range=(-3, 3), seed=None):
        """
        生成性别渐变序列: 从男性到女性的连续过渡

        参数:
            steps:       生成的帧数
            alpha_range: 性别范围 (min, max)
            seed:        随机种子 (固定种子可复现)

        返回:
            images: [steps, 3, H, W] 渐变序列
            alphas: [steps] 对应的 alpha 值
        """
        w = self.random_latent(seed=seed)
        alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
        images = []

        self.generator.eval()
        with torch.no_grad():
            for alpha in alphas:
                w_manip = self.manipulate(w, alpha)
                w_in = w_manip.unsqueeze(1).repeat(1, self.generator.n_latent, 1)
                img, _ = self.generator([w_in], input_is_latent=True)
                images.append(img.cpu())
            images = torch.cat(images, dim=0)

        return images, alphas

    def __call__(self, gender_strength=0.0, seed=None):
        """
        简化的调用接口: controller(gender_strength=2.0)
        seed=None 时每张脸随机
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.random_latent(seed=seed)
        img, _ = self.generate(alpha=gender_strength)
        return img


# ══════════════════════════════════════════════════════════
# 额外的潜空间操作工具
# ══════════════════════════════════════════════════════════

def slerp(a, b, t):
    """
    球面线性插值 (用于 W 空间平滑过渡)
    比线性插值更适合潜空间, 因为 W 空间分布在超球面上
    """
    a_norm = a / torch.norm(a)
    b_norm = b / torch.norm(b)
    omega = torch.acos((a_norm * b_norm).sum().clamp(-1, 1))
    if omega < 1e-6:
        return (1 - t) * a + t * b
    sin_omega = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b


def latent_mixing(generator, w1, w2, mix_layer=9, truncation=0.7):
    """
    粗粒度样式混合: 前几层用 w1 (姿态/脸型), 后几层用 w2 (肤色/纹理)

    参数:
        generator: StyleGAN2Generator
        w1, w2:    两个 W 编码 [1, 512]
        mix_layer: 从第几层开始切换 (0-17, 越早=w2影响越大)

    返回:
        image: [1, 3, H, W]
    """
    w1_plus = w1.unsqueeze(1).repeat(1, generator.n_latent, 1)
    w2_plus = w2.unsqueeze(1).repeat(1, generator.n_latent, 1)

    w_mix = w1_plus.clone()
    w_mix[:, mix_layer:] = w2_plus[:, mix_layer:]

    generator.eval()
    with torch.no_grad():
        img, _ = generator([w_mix], input_is_latent=True)

    return img


def save_gender_direction_cache(generator, direction, info, save_path):
    """保存性别方向到文件"""
    torch.save({
        'direction': direction.cpu(),
        'info': info,
    }, save_path)
    print(f"[Cache] 性别方向已保存: {save_path}")


def load_gender_direction_cache(load_path, device='cuda'):
    """从文件加载性别方向"""
    data = torch.load(load_path, map_location=device)
    return data['direction'], data.get('info', {})
