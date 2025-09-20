"""
训练数据管理器
提供科学的实验数据组织、存储和分析功能
"""

import json
import csv
import shutil
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataManager:
    """训练数据管理器"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self._ensure_directory_structure()
        
    def _ensure_directory_structure(self):
        """确保目录结构存在"""
        dirs = [
            self.base_dir / "experiments",
            self.base_dir / "experiments" / "archive",
            self.base_dir / "datasets",
            self.base_dir / "shared" / "environment_maps",
            self.base_dir / "shared" / "reference_models",
            self.base_dir / "shared" / "analysis_templates"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_experiment(
        self,
        algorithm: str,
        environment: str = "airsim",
        name: str = "",
        description: str = "",
        tags: List[str] = None,
        hyperparameters: Dict[str, Any] = None
    ) -> str:
        """
        创建新实验
        
        Returns:
            实验ID
        """
        
        # 生成实验ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_str = "_".join(tags) if tags else "experiment"
        experiment_id = f"{algorithm}_{environment}_{timestamp}_{tag_str}"
        
        # 创建实验目录
        exp_dir = self.base_dir / "experiments" / experiment_id
        
        # 创建子目录结构
        subdirs = [
            "config",
            "logs/tensorboard/train",
            "logs/tensorboard/eval", 
            "logs/tensorboard/system",
            "checkpoints",
            "metrics",
            "artifacts/videos",
            "artifacts/images",
            "artifacts/trajectories",
            "artifacts/analysis",
            "reports"
        ]
        
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # 创建实验元数据
        metadata = {
            "experiment_id": experiment_id,
            "name": name or f"{algorithm.upper()} Training",
            "description": description,
            "created_date": datetime.now().isoformat(),
            "status": "created",
            "duration": "00:00:00",
            "tags": tags or [],
            "algorithm": {
                "name": algorithm,
                "version": "modern_v1.0"
            },
            "environment": {
                "name": environment,
                "version": "1.8.1" if environment == "airsim" else "unknown"
            },
            "hyperparameters": hyperparameters or {},
            "results": {},
            "files": {
                "tensorboard": f"logs/tensorboard/",
                "metrics": "metrics/",
                "config": "config/"
            },
            "notes": ""
        }
        
        # 保存元数据
        metadata_path = exp_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验已创建: {experiment_id}")
        return experiment_id
    
    def get_experiment_path(self, experiment_id: str) -> Path:
        """获取实验路径"""
        return self.base_dir / "experiments" / experiment_id
    
    def get_tensorboard_path(self, experiment_id: str, log_type: str = "train") -> Path:
        """获取TensorBoard日志路径"""
        return self.get_experiment_path(experiment_id) / "logs" / "tensorboard" / log_type
    
    def get_checkpoints_path(self, experiment_id: str) -> Path:
        """获取检查点路径"""
        return self.get_experiment_path(experiment_id) / "checkpoints"
    
    def get_metrics_path(self, experiment_id: str) -> Path:
        """获取指标路径"""
        return self.get_experiment_path(experiment_id) / "metrics"
    
    def save_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, Any],
        metrics_type: str = "training"
    ):
        """保存指标数据"""
        metrics_dir = self.get_metrics_path(experiment_id)
        metrics_file = metrics_dir / f"{metrics_type}_metrics.json"
        
        # 如果文件已存在，加载现有数据
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}
        
        # 合并数据
        existing_data.update(metrics)
        
        # 保存更新后的数据
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    def save_episode_data(
        self,
        experiment_id: str,
        episode: int,
        reward: float,
        length: int,
        time_taken: float,
        additional_data: Dict[str, Any] = None
    ):
        """保存单个episode数据"""
        metrics_dir = self.get_metrics_path(experiment_id)
        csv_file = metrics_dir / "episode_rewards.csv"
        
        # 准备数据行
        episode_data = {
            "episode": episode,
            "reward": reward,
            "length": length,
            "time_taken": time_taken,
            "timestamp": datetime.now().isoformat()
        }
        
        if additional_data:
            episode_data.update(additional_data)
        
        # 写入CSV文件
        file_exists = csv_file.exists()
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=episode_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(episode_data)
    
    def save_loss_data(
        self,
        experiment_id: str,
        step: int,
        losses: Dict[str, float]
    ):
        """保存损失数据"""
        metrics_dir = self.get_metrics_path(experiment_id)
        csv_file = metrics_dir / "loss_curves.csv"
        
        # 准备数据行
        loss_data = {
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        loss_data.update(losses)
        
        # 写入CSV文件
        file_exists = csv_file.exists()
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=loss_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(loss_data)
    
    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        results: Dict[str, Any] = None,
        notes: str = ""
    ):
        """更新实验状态"""
        exp_dir = self.get_experiment_path(experiment_id)
        metadata_path = exp_dir / "metadata.json"
        
        if not metadata_path.exists():
            logger.error(f"实验元数据不存在: {experiment_id}")
            return
        
        # 加载现有元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 更新状态
        metadata["status"] = status
        if results:
            metadata["results"].update(results)
        if notes:
            metadata["notes"] = notes
        
        # 计算训练时长
        if status in ["completed", "failed", "stopped"]:
            created_time = datetime.fromisoformat(metadata["created_date"])
            duration = datetime.now() - created_time
            metadata["duration"] = str(duration).split('.')[0]  # 去掉微秒
        
        # 保存更新后的元数据
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def list_experiments(
        self,
        algorithm: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """列出实验"""
        exp_dir = self.base_dir / "experiments"
        experiments = []
        
        for exp_path in exp_dir.iterdir():
            if exp_path.is_dir() and exp_path.name != "archive":
                metadata_path = exp_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # 应用筛选条件
                    if algorithm and metadata.get("algorithm", {}).get("name") != algorithm:
                        continue
                    if status and metadata.get("status") != status:
                        continue
                    if tags and not any(tag in metadata.get("tags", []) for tag in tags):
                        continue
                    
                    experiments.append(metadata)
        
        # 按创建时间排序
        experiments.sort(key=lambda x: x.get("created_date", ""), reverse=True)
        
        if limit:
            experiments = experiments[:limit]
        
        return experiments
    
    def export_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """导出实验指标"""
        metrics_dir = self.get_metrics_path(experiment_id)
        exported_data = {}
        
        # 导出CSV数据
        csv_files = ["episode_rewards.csv", "loss_curves.csv"]
        for csv_file in csv_files:
            csv_path = metrics_dir / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                exported_data[csv_file.replace('.csv', '')] = df.to_dict('records')
        
        # 导出JSON数据
        json_files = metrics_dir.glob("*_metrics.json")
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                exported_data[json_file.stem] = data
        
        return exported_data
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """比较多个实验"""
        comparison_data = {
            "experiments": [],
            "summary": {},
            "metrics_comparison": {}
        }
        
        for exp_id in experiment_ids:
            # 加载实验元数据
            exp_dir = self.get_experiment_path(exp_id)
            metadata_path = exp_dir / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                comparison_data["experiments"].append(metadata)
                
                # 加载指标数据
                metrics = self.export_metrics(exp_id)
                comparison_data["metrics_comparison"][exp_id] = metrics
        
        # 生成对比总结
        if comparison_data["experiments"]:
            best_reward = max(
                exp.get("results", {}).get("best_reward", float('-inf'))
                for exp in comparison_data["experiments"]
            )
            comparison_data["summary"]["best_reward"] = best_reward
            comparison_data["summary"]["total_experiments"] = len(experiment_ids)
        
        return comparison_data
    
    def archive_experiment(self, experiment_id: str):
        """归档实验"""
        exp_dir = self.get_experiment_path(experiment_id)
        archive_dir = self.base_dir / "experiments" / "archive"
        
        if exp_dir.exists():
            # 移动到归档目录
            target_dir = archive_dir / experiment_id
            shutil.move(str(exp_dir), str(target_dir))
            logger.info(f"实验已归档: {experiment_id}")
        else:
            logger.error(f"实验不存在: {experiment_id}")
    
    def cleanup_old_experiments(self, days_threshold: int = 30):
        """清理旧实验"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        exp_dir = self.base_dir / "experiments"
        
        for exp_path in exp_dir.iterdir():
            if exp_path.is_dir() and exp_path.name != "archive":
                metadata_path = exp_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    created_date = datetime.fromisoformat(metadata["created_date"])
                    
                    # 跳过生产实验和重要标签
                    if "production" in metadata.get("tags", []):
                        continue
                    
                    if created_date < cutoff_date:
                        logger.info(f"清理旧实验: {exp_path.name}")
                        self.archive_experiment(exp_path.name)