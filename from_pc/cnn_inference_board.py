"""
板子端 CNN情绪分类推理脚本
============================
用途: 加载PC训练的DeepCNN模型, 对fiboaisdk生成的128x128热力图进行7类情绪推理

依赖: PyTorch (CPU), numpy
用法:
  python cnn_inference_board.py --model v9_cnn_deep_best.pth --heatmap test_heatmap.npy

输入:
  - 热力图文件: npy格式, shape=(3, 128, 128), float32, 值域[0,1]
    (由fiboaisdk提取468个landmark后生成, 与训练数据同分布)

输出:
  - 预测情绪标签(0~6) + 中文名 + 置信度
  - 7类概率分布

作者: AI | 日期: 2026-05-09 | 版本: V11-BoardDeploy
"""
import os, sys, json, argparse
import numpy as np

# ==================== PyTorch导入 (兼容板子CPU环境) ====================
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    print("[ERROR] PyTorch未安装! 请运行: pip install torch")
    TORCH_OK = False
    sys.exit(1)


# ==================== 类别定义 (与训练完全一致) ====================
LABEL_ORDER = ['惊讶', '恐惧', '厌恶', '开心', '悲伤', '愤怒', '中性']
NUM_CLASSES = 7
HEATMAP_SIZE = 128


# ==================== 模型定义 (必须与训练时的DeepEmotionCNN完全一致) ====================
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class DeepEmotionCNN(nn.Module):
    """
    DeepCNN: ~48万参数, 残差结构
    输入: (3, 128, 128) 热力图 → 输出: 7类logits
    """
    def __init__(self, num_classes=7, dropout_rate=0.5):  # 推理时dropout不影响结果
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128→64
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(24, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),  # 64→32
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ResidualBlock(48),
            nn.MaxPool2d(2),  # 32→16
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d(4),  # →4×4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate*0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# ==================== 推理引擎 ====================
class EmotionClassifier:
    """
    CNN情绪分类器封装
    
    使用方法:
        clf = EmotionClassifier('v9_cnn_deep_best.pth')
        result = clf.predict(heatmap_np)   # heatmap: (3,128,128) float32 [0,1]
        # result = {'label': 3, 'name': '开心', 'confidence': 0.72, 'probs': [...]}
    """

    def __init__(self, model_path, device='cpu'):
        """
        加载模型
        
        Args:
            model_path: .pth权重文件路径
            device: 'cpu'(板子) 或 'cuda'(有GPU时)
        """
        self.device = torch.device(device)
        self.label_order = LABEL_ORDER

        print(f"[推理] 设备: {self.device}")
        print(f"[推理] 加载模型: {model_path}")

        # 构建模型结构
        self.model = DeepEmotionCNN(num_classes=NUM_CLASSES, dropout_rate=0.0).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        params = self.model.get_num_params()
        self.best_acc = checkpoint.get('best_acc', '?')
        self.model_type = checkpoint.get('model_type', 'deep')

        print(f"[推理] 模型类型: {self.model_type.upper()} | 参数量: {params:,}")
        print(f"[推理] 训练时最佳ValAcc: {self.best_acc}%")
        print(f"[推理] 就绪 OK")

    @torch.no_grad()
    def predict(self, heatmap):
        """
        对单张热力图进行情绪预测

        Args:
            heatmap: numpy数组, shape=(3, H, W), float32, 值域[0,1]
                     H和W应为128 (会自动resize到HEATMAP_SIZE)

        Returns:
            dict: {
                'label': int,          # 预测类别 0~6
                'name': str,           # 中文名如'开心'
                'confidence': float,   # 置信度 0~1
                'probs': np.ndarray,   # 7类概率分布
                'all_results': list    # 所有类的详细信息
            }
        """
        # 预处理
        if isinstance(heatmap, str):
            # 如果传入的是文件路径
            data = np.load(heatmap, allow_pickle=True)
            if isinstance(data, dict):
                heatmap = data['heatmap'] if 'heatmap' in data else data['heatmaps'][0]
            else:
                heatmap = data

        heatmap = np.asarray(heatmap, dtype=np.float32)

        # 形状检查和调整
        if heatmap.ndim == 2:
            # 单通道 → 复制为3通道
            heatmap = np.stack([heatmap] * 3, axis=0)
        elif heatmap.ndim == 3 and heatmap.shape[0] not in (1, 3):
            # 可能是 (H,W,C) 格式
            if heatmap.shape[-1] == 3 or heatmap.shape[-1] == 1:
                heatmap = heatmap.transpose(2, 0, 1)
            elif heatmap.shape[0] == 128:  # 已经是 (C,H,W)
                pass
            else:
                raise ValueError(f"无法识别的热力图形状: {heatmap.shape}")

        # Resize到标准尺寸
        if heatmap.shape[1] != HEATMAP_SIZE or heatmap.shape[2] != HEATMAP_SIZE:
            # 简单双线性插值resize
            import scipy.ndimage
            new_hmap = np.zeros((3, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
            for c in range(3):
                zoom_factor = (HEATMAP_SIZE / heatmap.shape[1], HEATMAP_SIZE / heatmap.shape[2])
                new_hmap[c] = scipy.ndimage.zoom(heatmap[c], zoom_factor, order=1)
                new_hmap[c] = np.clip(new_hmap[c], 0, 1)
            heatmap = new_hmap

        # 裁剪到[0,1]
        heatmap = np.clip(heatmap, 0, 1)

        # 转tensor
        input_tensor = torch.from_numpy(heatmap).unsqueeze(0).to(self.device)  # (1,3,H,W)

        # 推理
        outputs = self.model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        # 结果整理
        pred_label = int(probs.argmax())
        confidence = float(probs[pred_label])

        all_results = []
        for i in range(NUM_CLASSES):
            all_results.append({
                'label': i,
                'name': LABEL_ORDER[i],
                'probability': round(float(probs[i]), 4),
            })
        # 按概率降序排列
        all_results.sort(key=lambda x: x['probability'], reverse=True)

        return {
            'label': pred_label,
            'name': LABEL_ORDER[pred_label],
            'confidence': round(confidence, 4),
            'probs': np.round(probs, 4),
            'all_results': all_results,
        }

    def predict_batch(self, heatmaps):
        """批量推理 (N张热力图)"""
        if isinstance(heatmaps, str):
            data = np.load(heatmaps, allow_pickle=True).item()
            heatmaps = data['heatmaps']

        heatmaps = np.clip(np.asarray(heatmaps, dtype=np.float32), 0, 1)
        input_tensor = torch.from_numpy(heatmaps).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        pred_labels = probs.argmax(axis=1)
        results = []
        for i in range(len(pred_labels)):
            results.append({
                'label': int(pred_labels[i]),
                'name': LABEL_ORDER[pred_labels[i]],
                'confidence': round(float(probs[i][pred_labels[i]]), 4),
            })

        return results


# ==================== 主入口 (命令行工具) ====================
def main():
    parser = argparse.ArgumentParser(description='CNN情绪分类推理 (板子端)')
    parser.add_argument('--model', '-m', type=str, default='v9_cnn_deep_best.pth',
                        help='模型权重文件路径')
    parser.add_argument('--heatmap', '-i', type=str, default=None,
                        help='热力图npy文件路径 (shape=(3,128,128))')
    parser.add_argument('--batch-dir', '-d', type=str, default=None,
                        help='批量推理: 包含多个npy的目录')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='推理设备 (默认cpu)')
    parser.add_argument('--json-out', type=str, default=None,
                        help='结果保存为JSON文件路径')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] 模型文件不存在: {args.model}")
        sys.exit(1)

    # 初始化分类器
    clf = EmotionClassifier(args.model, device=args.device)

    results_all = []

    if args.batch_dir:
        # 批量推理目录下所有npy
        npy_files = sorted([f for f in os.listdir(args.batch_dir) if f.endswith('.npy')])
        print(f"\n[批量] 发现 {len(npy_files)} 个热力图文件")
        
        for npy_file in npy_files:
            path = os.path.join(args.batch_dir, npy_file)
            try:
                r = clf.predict(path)
                results_all.append({'file': npy_file, **r})
                print(f"  {npy_file}: {r['name']} ({r['confidence']:.1%})")
            except Exception as e:
                print(f"  {npy_file}: ERROR - {e}")

    elif args.heatmap:
        # 单张推理
        print(f"\n[推理] 热力图: {args.heatmap}")
        result = clf.predict(args.heatmap)
        results_all = result

        print(f"\n{'═'*50}")
        print(f"★ 预测结果: {result['name']} (类别{result['label']})")
        print(f"  置信度: {result['confidence']:.1%}")
        print(f"{'─'*50}\n各类概率:")
        for item in result['all_results']:
            bar = "█" * int(item['probability'] * 30)
            print(f"  {item['name']:>4}({item['label']}): {item['probability']:.1%} {bar}")
    else:
        # 交互模式 / 测试模式
        print("\n[测试] 无指定输入, 运行自检...")
        
        # 生成一个假热力图测试模型是否正常工作
        fake_hm = np.random.rand(3, 128, 128).astype(np.float32) * 0.3
        result = clf.predict(fake_hm)
        
        print(f"\n  测试通过! 模型输出正常")
        print(f"  (随机输入预测: {result['name']}, 这是正常的)")
        print(f"\n提示: 用法示例:")
        print(f"  python {__file__} -m v9_cnn_deep_best.pth -i test_heatmap.npy")
        print(f"  python {__file__} -m v9_cnn_deep_best.pth -d ./heatmaps/")
        return

    # JSON输出
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(results_all, f, ensure_ascii=False, indent=2)
        print(f"\n[保存] 结果已写入: {args.json_out}")


if __name__ == '__main__':
    main()
