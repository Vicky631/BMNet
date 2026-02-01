"""
Training, evaluation and visualization functions
"""
import math
import os
from typing import Iterable
from PIL import Image
import numpy as np
from FSC_147 import FSC147DatasetLoader
# 导入CSV工具类
from csv_helper import EvaluationCSVWriter

import torch

EVAL_CFG = {
    "distance_thresh": 10.0,  # 定位匹配距离阈值
    "dsizensity_map_size": (512, 512),  # 密度图固定尺寸
    "annotation_path": '/mnt/mydisk/wjj/dataset/FSC_147/annotation_FSC_147_384.json',
    "image_root_path": '/mnt/mydisk/wjj/dataset/FSC_147/images_384_VarV2',
    "csv_save_dir": "/mnt/mydisk/wjj/BMNet/experiments/FSC147/eval_csv"  # CSV保存目录
}

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    loss_sum = 0
    loss_counting = 0
    loss_contrast = 0

    for idx, sample in enumerate(data_loader):
        img, patches, targets = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        patches['scale_embedding'] = patches['scale_embedding'].to(device)
        density_map = targets['density_map'].to(device)
        pt_map = targets['pt_map'].to(device)

        outputs = model(img, patches, is_train=True)

        dest = outputs['density_map']
        if epoch < 5: # check if training process get stucked in local optimal.
            print(dest.sum().item(), density_map.sum().item(), dest.sum().item()*10000 / (img.shape[-2] * img.shape[-1]))
        counting_loss, contrast_loss = criterion(outputs, density_map, pt_map)
        loss = counting_loss if isinstance(contrast_loss, int) else counting_loss + contrast_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            continue

        loss_sum += loss_value
        loss_contrast += contrast_loss if isinstance(contrast_loss, int) else contrast_loss.item()
        loss_counting += counting_loss.item()

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if (idx + 1) % 10 == 0:
            print('Epoch: %d, %d / %d, loss: %.8f, counting loss: %.8f, contrast loss: %.8f'%(epoch, idx+1,
                                                                                              len(data_loader),
                                                                                              loss_sum / (idx+1),
                                                                                              loss_counting / (idx+1),
                                                                                              loss_contrast / (idx+1)))

    return loss_sum / len(data_loader)

@torch.no_grad()
def evaluate(model, data_loader, device, output_dir):
    # 初始化基础指标累加器
    mae = 0.0
    mse = 0.0

    # 初始化样本级指标存储列表
    all_filenames = []
    all_gt_counts = []
    all_pred_counts = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    model.eval()

    # 初始化CSV写入器
    csv_writer = EvaluationCSVWriter(
        save_dir=EVAL_CFG["csv_save_dir"],
        filename_prefix="eval_metrics"
    )

    # 只初始化一次标注加载器，提升效率
    loader = FSC147DatasetLoader(
        annotation_file=EVAL_CFG["annotation_path"],
        image_root=EVAL_CFG["image_root_path"],
        scale_mode='none'
    )

    for idx, sample in enumerate(data_loader):
        img, patches, targets = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        patches['scale_embedding'] = patches['scale_embedding'].to(device)
        gtcount = targets['gtcount']

        outputs = model(img, patches, is_train=False)

        # ==================定位指标计算与收集======================
        filename = data_loader.dataset.data_list[idx]
        image_name = filename[0] if isinstance(filename, (list, tuple)) else filename
        result = loader.get_annotations(image_name, return_scaled=True, return_numpy=False)
        gt_points = result['points']

        # 初始化定位指标默认值（防止计算失败）
        f1 = 0.0
        precision = 0.0
        recall = 0.0

        try:
            from localization import evaluate_detection_metrics, get_pred_points_from_density
            density_map = outputs.squeeze(0).squeeze(0)
            f1, precision, recall = evaluate_detection_metrics(
                pred_density_map=density_map,
                gt_points=gt_points,
                distance_thresh=EVAL_CFG["distance_thresh"]
            )
            # 收集点信息（可选，如需保存可添加到CSV）
            pred_points = get_pred_points_from_density(density_map)
        except Exception as e:
            print(f"警告：计算{image_name}定位指标失败: {str(e)}")

        # ==================计数指标计算======================
        pred_count = outputs.sum().item()
        error = abs(pred_count - gtcount.item())
        mae += error
        mse += error ** 2

        # ==================收集样本级指标======================
        all_filenames.append(image_name)
        all_gt_counts.append(gtcount.item())
        all_pred_counts.append(pred_count)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)

        # 打印单样本指标（可选，便于实时监控）
        print(f"[{idx+1}/{len(data_loader)}] {image_name} - GT:{gtcount.item()}, Pred:{pred_count:.2f}, "
              f"Precision:{precision:.4f}, Recall:{recall:.4f}, F1:{f1:.4f}")

    # ==================计算全局平均指标======================
    total_samples = len(data_loader)
    avg_mae = mae / total_samples if total_samples > 0 else 0.0
    avg_mse = math.sqrt(mse / total_samples) if total_samples > 0 else 0.0

    # 计算定位指标的均值和标准差
    avg_precision = np.mean(all_precisions) if all_precisions else 0.0
    std_precision = np.std(all_precisions) if all_precisions else 0.0
    avg_recall = np.mean(all_recalls) if all_recalls else 0.0
    std_recall = np.std(all_recalls) if all_recalls else 0.0
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
    std_f1 = np.std(all_f1_scores) if all_f1_scores else 0.0

    # ==================保存指标到CSV======================
    # 1. 保存样本级指标
    sample_metrics = {
        "filenames": all_filenames,
        "gt_counts": all_gt_counts,
        "pred_counts": all_pred_counts,
        "precisions": all_precisions,
        "recalls": all_recalls,
        "f1_scores": all_f1_scores
    }
    csv_writer.save_sample_metrics(sample_metrics, EVAL_CFG["distance_thresh"])

    # 2. 保存汇总级指标
    summary_metrics = {
        "dataset_type": "evaluation",
        "total_samples": total_samples,
        "distance_thresh": EVAL_CFG["distance_thresh"],
        "avg_mae": avg_mae,
        "avg_rmse": avg_mse,
        "best_mae": avg_mae,  # 若需记录最佳值可后续扩展
        "best_rmse": avg_mse, # 若需记录最佳值可后续扩展
        "avg_precision": avg_precision,
        "std_precision": std_precision,
        "avg_recall": avg_recall,
        "std_recall": std_recall,
        "avg_f1": avg_f1,
        "std_f1": std_f1
    }
    csv_writer.save_summary_metrics(summary_metrics)

    # ==================保存到日志文件======================
    log_content = (
        f'MAE {avg_mae:.2f}, MSE {avg_mse:.2f} \n'
        f'Average Precision {avg_precision:.4f} (±{std_precision:.4f}), '
        f'Average Recall {avg_recall:.4f} (±{std_recall:.4f}), '
        f'Average F1 {avg_f1:.4f} (±{std_f1:.4f})\n'
    )

    with open(os.path.join(output_dir, 'result.txt'), 'a') as f:
        f.write(log_content)

    print(log_content)

    return avg_mae, avg_mse

@torch.no_grad()
def visualization(cfg, model, dataset, data_loader, device, output_dir):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    cmap = plt.cm.get_cmap('jet')
    visualization_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(visualization_dir, exist_ok=True)

    # 初始化基础指标累加器
    mae = 0.0
    mse = 0.0

    # 初始化样本级指标存储列表
    all_filenames = []
    all_gt_counts = []
    all_pred_counts = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    model.eval()

    # 初始化CSV写入器（可视化单独生成CSV，前缀区分）
    csv_writer = EvaluationCSVWriter(
        save_dir=EVAL_CFG["csv_save_dir"],
        filename_prefix="vis_metrics"
    )

    # 只初始化一次标注加载器
    loader = FSC147DatasetLoader(
        annotation_file=EVAL_CFG["annotation_path"],
        image_root=EVAL_CFG["image_root_path"],
        scale_mode='none'
    )

    for idx, sample in enumerate(data_loader):
        img, patches, targets = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        patches['scale_embedding'] = patches['scale_embedding'].to(device)
        gtcount = targets['gtcount']
        gt_density = targets['density_map']

        outputs = model(img, patches, is_train=False)

        # ==================计数指标计算======================
        pred_count = outputs.sum().item()
        error = abs(pred_count - gtcount.item())
        mae += error
        mse += error ** 2

        # ==================定位指标计算======================
        file_name = dataset.data_list[idx][0]
        # 初始化定位指标默认值
        f1 = 0.0
        precision = 0.0
        recall = 0.0

        try:
            from localization import evaluate_detection_metrics, get_pred_points_from_density
            result = loader.get_annotations(file_name, return_scaled=True, return_numpy=False)
            gt_points = result['points']
            density_map = outputs.squeeze(0).squeeze(0)

            f1, precision, recall = evaluate_detection_metrics(
                pred_density_map=density_map,
                gt_points=gt_points,
                distance_thresh=EVAL_CFG["distance_thresh"]
            )
        except Exception as e:
            print(f"警告：计算{file_name}定位指标失败: {str(e)}")

        # ==================收集样本级指标======================
        all_filenames.append(file_name)
        all_gt_counts.append(gtcount.item())
        all_pred_counts.append(pred_count)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)

        # ==================可视化逻辑（原有逻辑优化）======================
        file_path = os.path.join(dataset.data_dir, 'images_384_VarV2', file_name)
        origin_img = Image.open(file_path).convert("RGB")
        origin_img = np.array(origin_img)
        h, w, _ = origin_img.shape

        density_map = outputs
        density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()
        gt_density = torch.nn.functional.interpolate(gt_density, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()

        # 防止除零错误
        density_map_max = density_map.max() if density_map.max() > 0 else 1e-14
        gt_density_max = gt_density.max() if gt_density.max() > 0 else 1e-14

        density_map = cmap(density_map / density_map_max) * 255.0
        density_map = density_map[:,:,0:3] * 0.5 + origin_img * 0.5
        gt_density = cmap(gt_density / gt_density_max) * 255.0
        gt_density = gt_density[:,:,0:3] * 0.5 + origin_img * 0.5

        fig = plt.figure(dpi=800)
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.axis('off')
        ax2.axis('off')

        ax1.set_title(f"GT: {gtcount.item()}")
        ax2.set_title(f"Pred: {pred_count:.2f}")
        ax1.imshow(gt_density.astype(np.uint8))
        ax2.imshow(density_map.astype(np.uint8))

        save_path = os.path.join(visualization_dir, os.path.basename(file_name))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 打印进度
        print(f"可视化进度: [{idx+1}/{len(data_loader)}] {file_name} 完成")

    # ==================计算全局平均指标======================
    total_samples = len(data_loader)
    avg_mae = mae / total_samples if total_samples > 0 else 0.0
    avg_mse = math.sqrt(mse / total_samples) if total_samples > 0 else 0.0

    # 计算定位指标的均值和标准差
    avg_precision = np.mean(all_precisions) if all_precisions else 0.0
    std_precision = np.std(all_precisions) if all_precisions else 0.0
    avg_recall = np.mean(all_recalls) if all_recalls else 0.0
    std_recall = np.std(all_recalls) if all_recalls else 0.0
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
    std_f1 = np.std(all_f1_scores) if all_f1_scores else 0.0

    # ==================保存指标到CSV======================
    # 1. 保存样本级指标
    sample_metrics = {
        "filenames": all_filenames,
        "gt_counts": all_gt_counts,
        "pred_counts": all_pred_counts,
        "precisions": all_precisions,
        "recalls": all_recalls,
        "f1_scores": all_f1_scores
    }
    csv_writer.save_sample_metrics(sample_metrics, EVAL_CFG["distance_thresh"])

    # 2. 保存汇总级指标
    summary_metrics = {
        "dataset_type": "visualization",
        "total_samples": total_samples,
        "distance_thresh": EVAL_CFG["distance_thresh"],
        "avg_mae": avg_mae,
        "avg_rmse": avg_mse,
        "best_mae": avg_mae,
        "best_rmse": avg_mse,
        "avg_precision": avg_precision,
        "std_precision": std_precision,
        "avg_recall": avg_recall,
        "std_recall": std_recall,
        "avg_f1": avg_f1,
        "std_f1": std_f1
    }
    csv_writer.save_summary_metrics(summary_metrics)

    # ==================保存到日志文件======================
    log_content = (
        f'[Visualization] MAE {avg_mae:.2f}, MSE {avg_mse:.2f} \n'
        f'[Visualization] Average Precision {avg_precision:.4f} (±{std_precision:.4f}), '
        f'Average Recall {avg_recall:.4f} (±{std_recall:.4f}), '
        f'Average F1 {avg_f1:.4f} (±{std_f1:.4f})\n'
    )

    with open(os.path.join(output_dir, 'result.txt'), 'a') as f:
        f.write(log_content)

    print(log_content)

    return avg_mae, avg_mse