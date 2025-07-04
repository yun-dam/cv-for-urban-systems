"""
独立模型评估脚本 - Stanford UGVR CV项目
用于评估训练好的CLIPSeg模型在测试集上的性能
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from utils import (
    FineTuneDataset, 
    create_data_loader, 
    load_model, 
    calculate_iou,
    evaluate_model
)

def parse_args():
    """解析命令行参数（使用config.py中的默认值）"""
    parser = argparse.ArgumentParser(description='CLIPSeg模型评估脚本')
    
    # 使用配置文件中的默认值
    parser.add_argument('--model_path', type=str, 
                        default=str(EVALUATION_CONFIG['default_model_path']),
                        help='训练好的模型路径')
    parser.add_argument('--test_data', type=str, 
                        default=str(EVALUATION_CONFIG['default_test_data']),
                        help='测试数据目录路径')
    parser.add_argument('--output_dir', type=str, 
                        default=str(EVALUATION_CONFIG['default_output_dir']),
                        help='评估结果输出目录')
    parser.add_argument('--batch_size', type=int, 
                        default=EVALUATION_CONFIG['batch_size'],
                        help='批处理大小')
    parser.add_argument('--num_samples', type=int, 
                        default=EVALUATION_CONFIG['num_visualization_samples'],
                        help='可视化样本数量')
    parser.add_argument('--device', type=str, 
                        default=EVALUATION_CONFIG['device'],
                        help='计算设备 (auto/cpu/cuda/mps)')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """设置计算设备"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"使用设备: {device}")
    return device

def load_test_data(test_data_dir: str, batch_size: int, processor) -> Tuple[DataLoader, int]:
    """加载测试数据"""
    test_images = sorted(Path(test_data_dir).glob('*.tif'))
    test_images = [str(img) for img in test_images]
    
    if not test_images:
        raise ValueError(f"在 {test_data_dir} 中未找到测试图像")
    
    # 创建数据加载器
    test_loader = create_data_loader(
        test_images, 
        str(VAIHINGEN_LABELS_DIR), 
        URBAN_CLASSES, 
        processor,
        batch_size,
        shuffle=False
    )
    
    print(f"加载了 {len(test_images)} 张测试图像")
    return test_loader, len(test_images)

def evaluate_detailed_performance(model, test_loader, device, classes) -> Dict:
    """详细性能评估"""
    model.eval()
    all_predictions = []
    all_targets = []
    class_ious = {cls: [] for cls in classes}
    
    print("开始详细评估...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估进度"):
            # 模型预测
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            logits = outputs.logits
            labels = batch["labels"].to(device)
            
            # 确保logits和labels的维度匹配
            if logits.dim() == 3 and labels.dim() == 4:
                logits = logits.unsqueeze(1)
            elif logits.dim() == 4 and labels.dim() == 4:
                pass  # 维度已经匹配
            else:
                print(f"警告: logits形状={logits.shape}, labels形状={labels.shape}")
                
            predictions = torch.sigmoid(logits)
            
            # 转换为numpy数组
            pred_np = predictions.cpu().numpy()
            target_np = labels.cpu().numpy()
            
            # 计算每个类别的IoU
            for i, cls in enumerate(classes):
                pred_binary = (pred_np[:, i] > 0.5).astype(int)
                target_binary = target_np[:, i].astype(int)
                
                # 计算IoU
                iou = calculate_iou(pred_binary, target_binary)
                class_ious[cls].append(iou)
                
                # 收集预测和目标用于混淆矩阵
                all_predictions.extend(pred_binary.flatten())
                all_targets.extend(target_binary.flatten())
    
    # 计算平均IoU
    mean_ious = {}
    for cls in classes:
        mean_ious[cls] = np.mean(class_ious[cls]) if class_ious[cls] else 0.0
    
    # 总体mIoU
    mean_iou = np.mean(list(mean_ious.values()))
    
    results = {
        'class_ious': mean_ious,
        'mean_iou': mean_iou,
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    return results

def generate_performance_report(results: Dict, classes: List[str], output_dir: Path):
    """生成性能报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. IoU结果表格
    iou_df = pd.DataFrame([
        {'Class': cls, 'IoU': iou} 
        for cls, iou in results['class_ious'].items()
    ])
    iou_df.loc[len(iou_df)] = {'Class': 'Mean', 'IoU': results['mean_iou']}
    
    # 保存IoU结果
    iou_df.to_csv(output_dir / 'iou_results.csv', index=False)
    
    # 2. 混淆矩阵
    cm = confusion_matrix(results['targets'], results['predictions'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    
    # 3. 类别性能可视化
    plt.figure(figsize=(12, 8))
    bars = plt.bar(results['class_ious'].keys(), results['class_ious'].values())
    plt.axhline(y=results['mean_iou'], color='red', linestyle='--', 
                label=f'Mean IoU: {results["mean_iou"]:.3f}')
    plt.title('各类别IoU性能')
    plt.ylabel('IoU分数')
    plt.xlabel('类别')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'class_performance.png', dpi=300)
    plt.close()
    
    # 4. 详细报告
    report = {
        'evaluation_summary': {
            'mean_iou': results['mean_iou'],
            'class_ious': results['class_ious'],
            'total_samples': len(results['targets'])
        },
        'evaluation_time': pd.Timestamp.now().isoformat()
    }
    
    with open(output_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def visualize_predictions(model_path: str, test_data_dir: str, device, classes, 
                         output_dir: Path, num_samples: int = 10):
    """可视化预测结果 - 参考clipseg_run.py的4面板可视化方式"""
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    import tifffile
    
    # 加载预训练和微调模型用于对比
    # 加载预训练模型
    processor_pretrained = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)
    model_pretrained = CLIPSegForImageSegmentation.from_pretrained(PRETRAINED_MODEL)
    model_pretrained.to(device)
    model_pretrained.eval()
    
    # 加载微调模型
    processor_finetuned = CLIPSegProcessor.from_pretrained(model_path)
    model_finetuned = CLIPSegForImageSegmentation.from_pretrained(model_path)
    model_finetuned.to(device)
    model_finetuned.eval()
    
    # 分割颜色映射
    SEGMENTATION_COLORS_RGB = {
        "impervious surface": (255, 255, 255),  # White
        "building":           (0, 0, 255),      # Blue
        "low vegetation":     (0, 255, 255),    # Cyan
        "tree":               (0, 255, 0),      # Green
        "car":                (255, 255, 0),    # Yellow
        "background":         (255, 0, 0)       # Red
    }
    
    # 获取测试图像
    test_images = sorted(Path(test_data_dir).glob('*.tif'))
    if len(test_images) > num_samples:
        import random
        random.seed(42)
        test_images = random.sample(test_images, num_samples)
    
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"生成 {len(test_images)} 个可视化样本...")
    
    for i, image_path in enumerate(tqdm(test_images, desc="生成可视化")):
        # 对应的标签文件
        label_path = VAIHINGEN_LABELS_DIR / image_path.name
        
        if not Path(label_path).exists():
            print(f"标签文件不存在: {label_path}")
            continue
            
        try:
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            images = [image for _ in classes]
            
            # 预训练模型推理
            inputs_pretrained = processor_pretrained(images=images, text=classes, return_tensors="pt", padding=True)
            inputs_pretrained = {k: v.to(device) for k, v in inputs_pretrained.items()}
            with torch.no_grad():
                outputs_pretrained = model_pretrained(**inputs_pretrained)
            masks_pretrained = outputs_pretrained.logits.sigmoid().squeeze().cpu().numpy()
            resized_masks_pretrained = [
                np.array(Image.fromarray(m).resize(image.size, resample=Image.BILINEAR))
                for m in masks_pretrained
            ]
            
            # 微调模型推理
            inputs_finetuned = processor_finetuned(images=images, text=classes, return_tensors="pt", padding=True)
            inputs_finetuned = {k: v.to(device) for k, v in inputs_finetuned.items()}
            with torch.no_grad():
                outputs_finetuned = model_finetuned(**inputs_finetuned)
            masks_finetuned = outputs_finetuned.logits.sigmoid().squeeze().cpu().numpy()
            resized_masks_finetuned = [
                np.array(Image.fromarray(m).resize(image.size, resample=Image.BILINEAR))
                for m in masks_finetuned
            ]
            
            # 生成分割图
            color_map_rgb = np.array([SEGMENTATION_COLORS_RGB[prompt] for prompt in classes], dtype=np.uint8)
            
            # 预训练分割图
            stacked_masks_pretrained = np.stack(resized_masks_pretrained, axis=0)
            pred_labels_pretrained = np.argmax(stacked_masks_pretrained, axis=0)
            segmentation_map_pretrained = color_map_rgb[pred_labels_pretrained]
            
            # 微调分割图
            stacked_masks_finetuned = np.stack(resized_masks_finetuned, axis=0)
            pred_labels_finetuned = np.argmax(stacked_masks_finetuned, axis=0)
            segmentation_map_finetuned = color_map_rgb[pred_labels_finetuned]
            
            # 加载真实标签
            gt_img_bgr = tifffile.imread(label_path)
            
            # 创建4面板可视化 - 使用配置中的设置
            fig, axes = plt.subplots(2, 2, 
                                   figsize=EVALUATION_CONFIG['figure_size'], 
                                   dpi=EVALUATION_CONFIG['visualization_dpi'])
            
            # 原始图像
            axes[0, 0].imshow(np.array(image))
            axes[0, 0].set_title(f"Original Image\n{image_path.name}", fontsize=12)
            axes[0, 0].axis("off")
            
            # 预训练模型输出
            axes[0, 1].imshow(segmentation_map_pretrained)
            axes[0, 1].set_title(f"Pretrained CLIPSeg Output", fontsize=12)
            axes[0, 1].axis("off")
            
            # 微调模型输出
            axes[1, 0].imshow(segmentation_map_finetuned)
            axes[1, 0].set_title(f"Finetuned CLIPSeg Output", fontsize=12)
            axes[1, 0].axis("off")
            
            # 真实标签
            axes[1, 1].imshow(gt_img_bgr)
            axes[1, 1].set_title("Ground Truth", fontsize=12)
            axes[1, 1].axis("off")
            
            plt.tight_layout(pad=2.0)
            save_path = vis_dir / f"{image_path.stem}_4panel_comparison.png"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"处理图像 {image_path.name} 时发生错误: {e}")
            continue
    
    print(f"可视化结果已保存到: {vis_dir}")

def main():
    """主函数"""
    args = parse_args()
    
    # 显示配置信息
    print("🔧 评估配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  测试数据: {args.test_data}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  批处理大小: {args.batch_size}")
    print(f"  可视化样本数: {args.num_samples}")
    print()
    
    # 设置设备
    device = setup_device(args.device)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"评估结果将保存到: {output_dir}")
    
    try:
        # 加载模型
        print("加载模型...")
        model, processor, metadata = load_model(Path(args.model_path), device)
        
        # 加载测试数据
        print("加载测试数据...")
        test_loader, num_images = load_test_data(args.test_data, args.batch_size, processor)
        
        # 详细性能评估
        print("开始性能评估...")
        results = evaluate_detailed_performance(model, test_loader, device, URBAN_CLASSES)
        
        # 生成报告
        print("生成评估报告...")
        report = generate_performance_report(results, URBAN_CLASSES, output_dir)
        
        # 可视化预测结果
        print("生成可视化结果...")
        visualize_predictions(args.model_path, args.test_data, device, URBAN_CLASSES, 
                             output_dir, args.num_samples)
        
        # 打印结果摘要
        print("\n" + "="*50)
        print("评估结果摘要:")
        print("="*50)
        print(f"总体mIoU: {results['mean_iou']:.4f}")
        print("\n各类别IoU:")
        for cls, iou in results['class_ious'].items():
            print(f"  {cls}: {iou:.4f}")
        
        print(f"\n详细结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())