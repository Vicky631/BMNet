# utils/csv_helper.py
import os
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Union


class EvaluationCSVWriter:
    """
    评估指标CSV保存工具类（可复用）
    功能：将计数/定位指标保存为结构化CSV文件，支持追加/新建，便于后期对比分析
    """

    def __init__(self, save_dir: str, filename_prefix: str = "eval_metrics"):
        """
        初始化CSV写入器
        Args:
            save_dir: CSV文件保存目录
            filename_prefix: CSV文件前缀，最终文件名格式：{prefix}_{时间戳}.csv
        """
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix
        # 创建保存目录（确保存在）
        os.makedirs(self.save_dir, exist_ok=True)

        # 生成带时间戳的文件名（避免重复）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.save_dir, f"{filename_prefix}_{timestamp}.csv")

        # 初始化CSV表头
        self.sample_header = [
            "filename", "gt_count", "pred_count",
            "precision", "recall", "f1_score", "distance_thresh"
        ]
        self.summary_header = [
            "dataset_type", "total_samples", "distance_thresh",
            "avg_mae", "avg_rmse", "best_mae", "best_rmse",
            "avg_precision", "std_precision",
            "avg_recall", "std_recall",
            "avg_f1", "std_f1"
        ]

    def save_sample_metrics(self, sample_metrics: Dict, distance_thresh: float):
        """
        保存样本级详细指标到CSV
        Args:
            sample_metrics: 样本指标字典，需包含keys: filenames, gt_counts, pred_counts, precisions, recalls, f1_scores
            distance_thresh: 定位匹配距离阈值
        """
        # 检查输入完整性
        required_keys = ["filenames", "gt_counts", "pred_counts", "precisions", "recalls", "f1_scores"]
        for key in required_keys:
            if key not in sample_metrics:
                raise ValueError(f"sample_metrics缺少必要键：{key}")

        # 写入样本级数据
        with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.sample_header)
            writer.writeheader()

            # 逐样本写入
            for i in range(len(sample_metrics["filenames"])):
                row = {
                    "filename": sample_metrics["filenames"][i],
                    "gt_count": round(sample_metrics["gt_counts"][i], 1),
                    "pred_count": round(sample_metrics["pred_counts"][i], 1),
                    "precision": round(sample_metrics["precisions"][i], 4),
                    "recall": round(sample_metrics["recalls"][i], 4),
                    "f1_score": round(sample_metrics["f1_scores"][i], 4),
                    "distance_thresh": distance_thresh
                }
                writer.writerow(row)

        print(f"样本级指标已保存至: {self.csv_path}")

    def save_summary_metrics(self, summary_metrics: Dict):
        """
        追加保存全局汇总指标到CSV（单独的汇总文件，便于对比）
        Args:
            summary_metrics: 汇总指标字典，需包含self.summary_header的所有键
        """
        # 生成汇总文件路径（与样本文件同目录，后缀为_summary）
        summary_csv_path = self.csv_path.replace(".csv", "_summary.csv")

        # 写入汇总数据
        with open(summary_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.summary_header)
            writer.writeheader()
            # 格式化数值
            formatted_row = {}
            for key, value in summary_metrics.items():
                if isinstance(value, (int, float)):
                    formatted_row[key] = round(value, 4) if "std" in key or "avg" in key else round(value, 2)
                else:
                    formatted_row[key] = value
            writer.writerow(formatted_row)

        print(f"全局汇总指标已保存至: {summary_csv_path}")

    @staticmethod
    def load_metrics(csv_path: str) -> List[Dict]:
        """
        加载CSV中的指标（复用方法：用于其他代码读取对比）
        Args:
            csv_path: CSV文件路径
        Returns:
            指标列表，每个元素为一行的字典
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在：{csv_path}")

        metrics = []
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 数值类型转换
                for key, value in row.items():
                    if key in ["gt_count", "pred_count", "precision", "recall", "f1_score",
                               "distance_thresh", "avg_mae", "avg_rmse", "best_mae", "best_rmse",
                               "avg_precision", "std_precision", "avg_recall", "std_recall", "avg_f1", "std_f1"]:
                        row[key] = float(value)
                metrics.append(row)
        return metrics