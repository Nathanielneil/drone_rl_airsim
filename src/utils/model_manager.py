"""
现代化模型管理器
提供科学的模型保存、加载和版本管理功能
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """现代化模型管理器"""
    
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def save_model(
        self,
        model_path: str,
        config: Dict[str, Any],
        algorithm: str,
        experiment_name: str,
        performance_metrics: Dict[str, Any],
        model_type: str = "development",  # "development" or "production"
        version: Optional[str] = None,
        tag: Optional[str] = None,
        notes: str = ""
    ) -> Path:
        """
        保存模型到科学的目录结构中
        
        Args:
            model_path: 模型文件路径
            config: 训练配置
            algorithm: 算法名称 (sac, ppo等)
            experiment_name: 实验名称
            performance_metrics: 性能指标
            model_type: 模型类型 ("development" 或 "production")
            version: 版本号 (仅用于production)
            tag: 标签 (best, stable等)
            notes: 备注信息
            
        Returns:
            保存的模型目录路径
        """
        
        # 创建算法目录
        algorithm_dir = self.base_dir / algorithm
        algorithm_dir.mkdir(exist_ok=True)
        
        # 创建类型目录
        type_dir = algorithm_dir / model_type
        type_dir.mkdir(exist_ok=True)
        
        # 生成模型目录名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if model_type == "production":
            if not version:
                raise ValueError("Production模型必须指定version")
            date_str = datetime.now().strftime("%Y%m%d")
            if tag:
                model_dir_name = f"v{version}_{date_str}_{tag}"
            else:
                model_dir_name = f"v{version}_{date_str}"
        else:
            model_dir_name = f"exp_{experiment_name}_{timestamp}"
        
        model_dir = type_dir / model_dir_name
        model_dir.mkdir(exist_ok=True)
        
        # 保存模型文件
        target_model_path = model_dir / "model.pth"
        if Path(model_path).exists():
            shutil.copy2(model_path, target_model_path)
        
        # 保存配置
        config_path = model_dir / "config.yaml"
        if hasattr(config, 'save'):
            config.save(config_path)
        else:
            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 创建元数据
        metadata = {
            "model_name": f"{algorithm}_{model_dir_name}",
            "algorithm": algorithm,
            "created_date": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "model_type": model_type,
            "total_timesteps": performance_metrics.get("total_timesteps", 0),
            "final_reward": performance_metrics.get("final_reward", 0.0),
            "best_reward": performance_metrics.get("best_reward", 0.0),
            "episodes_trained": performance_metrics.get("episodes_trained", 0),
            "training_duration": performance_metrics.get("training_duration", "00:00:00"),
            "environment": "AirSim-v1.8.1",
            "gpu_used": performance_metrics.get("gpu_name", "Unknown"),
            "cuda_version": performance_metrics.get("cuda_version", "Unknown"),
            "tags": [tag] if tag else [],
            "notes": notes
        }
        
        if version:
            metadata["version"] = version
        
        # 保存元数据
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 保存性能指标
        metrics_path = model_dir / "performance_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, indent=2, ensure_ascii=False)
        
        # 如果是production模型，更新latest链接
        if model_type == "production":
            self._update_latest_link(algorithm, model_dir_name)
        
        logger.info(f"模型已保存到: {model_dir}")
        return model_dir
    
    def _update_latest_link(self, algorithm: str, model_dir_name: str):
        """更新latest软链接"""
        try:
            production_dir = self.base_dir / algorithm / "production"
            latest_link = production_dir / "latest"
            
            # 删除旧链接
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            
            # 创建新链接
            target = Path(model_dir_name)
            latest_link.symlink_to(target)
            logger.info(f"Latest链接已更新: {latest_link} -> {target}")
            
        except Exception as e:
            logger.warning(f"无法创建latest链接: {e}")
    
    def load_model(self, algorithm: str, model_identifier: str = "latest") -> Dict[str, Any]:
        """
        加载模型
        
        Args:
            algorithm: 算法名称
            model_identifier: 模型标识符 ("latest", 版本号, 或完整路径)
            
        Returns:
            包含模型路径、配置和元数据的字典
        """
        
        if model_identifier == "latest":
            model_path = self.base_dir / algorithm / "production" / "latest" / "model.pth"
            model_dir = self.base_dir / algorithm / "production" / "latest"
        else:
            # 尝试作为完整路径
            model_path = Path(model_identifier)
            if not model_path.exists():
                # 尝试在production中查找
                model_path = self.base_dir / algorithm / "production" / model_identifier / "model.pth"
                model_dir = self.base_dir / algorithm / "production" / model_identifier
                
                if not model_path.exists():
                    # 尝试在development中查找
                    model_path = self.base_dir / algorithm / "development" / model_identifier / "model.pth"
                    model_dir = self.base_dir / algorithm / "development" / model_identifier
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        
        # 加载元数据
        metadata_path = model_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # 加载配置
        config_path = model_dir / "config.yaml"
        config = {}
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        return {
            "model_path": model_path,
            "model_dir": model_dir,
            "metadata": metadata,
            "config": config
        }
    
    def list_models(self, algorithm: str, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出指定算法的所有模型
        
        Args:
            algorithm: 算法名称
            model_type: 模型类型筛选 ("production", "development", None为全部)
            
        Returns:
            模型信息列表
        """
        
        algorithm_dir = self.base_dir / algorithm
        if not algorithm_dir.exists():
            return []
        
        models = []
        
        # 搜索目录
        search_dirs = []
        if model_type is None:
            search_dirs = ["production", "development"]
        else:
            search_dirs = [model_type]
        
        for type_dir_name in search_dirs:
            type_dir = algorithm_dir / type_dir_name
            if not type_dir.exists():
                continue
                
            for model_dir in type_dir.iterdir():
                if model_dir.is_dir() and model_dir.name != "latest":
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            metadata["model_dir"] = str(model_dir)
                            models.append(metadata)
        
        # 按创建时间排序
        models.sort(key=lambda x: x.get("created_date", ""), reverse=True)
        return models
    
    def archive_model(self, algorithm: str, model_identifier: str):
        """将模型移动到archive目录"""
        # 实现模型存档逻辑
        pass
    
    def cleanup_old_models(self, algorithm: str, keep_count: int = 10):
        """清理旧的development模型，只保留最新的几个"""
        # 实现清理逻辑
        pass