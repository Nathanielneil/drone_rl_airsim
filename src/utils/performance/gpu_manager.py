"""
GPU性能管理器
针对 Windows 10 + CUDA 12.1 优化
提供GPU内存管理、性能监控、自动优化功能
"""

import os
import time
import logging
import threading
import subprocess
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import psutil

import torch
import numpy as np
from collections import deque, defaultdict

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml未安装，GPU监控功能受限")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil未安装，部分GPU功能不可用")

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryInfo:
    """GPU内存信息"""
    total: float  # GB
    used: float   # GB
    free: float   # GB
    utilization: float  # %
    temperature: float  # °C
    power_usage: float  # W


@dataclass
class PerformanceMetrics:
    """性能指标"""
    avg_fps: float
    avg_training_time: float
    gpu_memory_efficiency: float
    cpu_usage: float
    ram_usage: float
    gpu_utilization: float


class GPUMemoryManager:
    """GPU内存管理器"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = torch.cuda.is_available() and "cuda" in str(self.device)
        
        # 内存策略
        self.memory_fraction = 0.9  # 使用90%的GPU内存
        self.allow_growth = True
        self.clear_cache_interval = 100  # 每100步清理一次缓存
        self.step_count = 0
        
        # 内存使用历史
        self.memory_history = deque(maxlen=1000)
        self.peak_memory = 0.0
        
        if self.is_cuda:
            self._setup_cuda_memory()
            logger.info(f"GPU内存管理器初始化完成: {torch.cuda.get_device_name()}")
    
    def _setup_cuda_memory(self):
        """设置CUDA内存管理"""
        if not self.is_cuda:
            return
        
        # 设置内存分配器
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # 启用内存池
        torch.cuda.empty_cache()
        
        # 设置内存增长策略
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
    
    def get_memory_info(self) -> GPUMemoryInfo:
        """获取GPU内存信息"""
        if not self.is_cuda:
            return GPUMemoryInfo(0, 0, 0, 0, 0, 0)
        
        try:
            # PyTorch内存信息
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            
            # GPU属性
            props = torch.cuda.get_device_properties(self.device)
            total_memory = props.total_memory / 1024**3
            
            # 利用率和温度（如果可用）
            utilization = 0.0
            temperature = 0.0
            power_usage = 0.0
            
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                    
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temperature = temp
                    
                    power_info = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_usage = power_info / 1000.0  # mW to W
                except Exception as e:
                    logger.debug(f"NVML查询失败: {e}")
            
            info = GPUMemoryInfo(
                total=total_memory,
                used=allocated,
                free=total_memory - reserved,
                utilization=utilization,
                temperature=temperature,
                power_usage=power_usage
            )
            
            # 更新历史记录
            self.memory_history.append(allocated)
            self.peak_memory = max(self.peak_memory, allocated)
            
            return info
            
        except Exception as e:
            logger.error(f"获取GPU内存信息失败: {e}")
            return GPUMemoryInfo(0, 0, 0, 0, 0, 0)
    
    def optimize_memory(self) -> bool:
        """优化GPU内存使用"""
        if not self.is_cuda:
            return False
        
        try:
            # 清理缓存
            torch.cuda.empty_cache()
            
            # 检查内存碎片
            info = self.get_memory_info()
            memory_efficiency = info.used / info.total if info.total > 0 else 0
            
            if memory_efficiency > 0.9:
                logger.warning(f"GPU内存使用率过高: {memory_efficiency:.1%}")
                # 强制垃圾回收
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"内存优化失败: {e}")
            return False
    
    def auto_clear_cache(self):
        """自动清理缓存"""
        self.step_count += 1
        if self.step_count % self.clear_cache_interval == 0:
            if self.is_cuda:
                torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict:
        """获取内存统计信息"""
        if not self.is_cuda:
            return {}
        
        info = self.get_memory_info()
        
        return {
            "current_memory_gb": info.used,
            "peak_memory_gb": self.peak_memory,
            "memory_utilization": info.used / info.total if info.total > 0 else 0,
            "avg_memory_gb": np.mean(self.memory_history) if self.memory_history else 0,
            "memory_efficiency": len([m for m in self.memory_history if m > 0]) / len(self.memory_history) if self.memory_history else 0,
            "total_memory_gb": info.total,
            "gpu_utilization": info.utilization,
            "temperature_c": info.temperature,
            "power_usage_w": info.power_usage,
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitor_interval: float = 1.0):
        self.monitor_interval = monitor_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 性能数据
        self.fps_history = deque(maxlen=1000)
        self.training_times = deque(maxlen=1000)
        self.gpu_utilization = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)
        self.ram_usage = deque(maxlen=1000)
        
        # GPU管理器
        self.gpu_manager = GPUMemoryManager()
        
        # 性能阈值
        self.performance_thresholds = {
            "max_gpu_memory": 0.9,  # 90%
            "max_cpu_usage": 0.8,   # 80%
            "max_ram_usage": 0.8,   # 80%
            "min_fps": 10.0,        # 10 FPS
            "max_temperature": 85.0, # 85°C
        }
        
        logger.info("性能监控器初始化完成")
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集性能数据
                self._collect_performance_data()
                
                # 检查性能阈值
                self._check_performance_thresholds()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"性能监控错误: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_performance_data(self):
        """收集性能数据"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_usage.append(cpu_percent)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        ram_percent = memory.percent
        self.ram_usage.append(ram_percent)
        
        # GPU信息
        gpu_info = self.gpu_manager.get_memory_info()
        self.gpu_utilization.append(gpu_info.utilization)
    
    def _check_performance_thresholds(self):
        """检查性能阈值"""
        if not self.cpu_usage or not self.ram_usage or not self.gpu_utilization:
            return
        
        # 检查CPU使用率
        current_cpu = self.cpu_usage[-1]
        if current_cpu > self.performance_thresholds["max_cpu_usage"] * 100:
            logger.warning(f"CPU使用率过高: {current_cpu:.1f}%")
        
        # 检查内存使用率
        current_ram = self.ram_usage[-1]
        if current_ram > self.performance_thresholds["max_ram_usage"] * 100:
            logger.warning(f"内存使用率过高: {current_ram:.1f}%")
        
        # 检查GPU
        gpu_info = self.gpu_manager.get_memory_info()
        if gpu_info.utilization > self.performance_thresholds["max_gpu_memory"] * 100:
            logger.warning(f"GPU内存使用率过高: {gpu_info.utilization:.1f}%")
        
        if gpu_info.temperature > self.performance_thresholds["max_temperature"]:
            logger.warning(f"GPU温度过高: {gpu_info.temperature:.1f}°C")
    
    def record_fps(self, fps: float):
        """记录FPS"""
        self.fps_history.append(fps)
    
    def record_training_time(self, training_time: float):
        """记录训练时间"""
        self.training_times.append(training_time)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0.0
        avg_training_time = np.mean(self.training_times) if self.training_times else 0.0
        avg_gpu_util = np.mean(self.gpu_utilization) if self.gpu_utilization else 0.0
        avg_cpu_usage = np.mean(self.cpu_usage) if self.cpu_usage else 0.0
        avg_ram_usage = np.mean(self.ram_usage) if self.ram_usage else 0.0
        
        # GPU内存效率
        gpu_stats = self.gpu_manager.get_memory_stats()
        gpu_memory_efficiency = gpu_stats.get("memory_efficiency", 0.0)
        
        return PerformanceMetrics(
            avg_fps=avg_fps,
            avg_training_time=avg_training_time,
            gpu_memory_efficiency=gpu_memory_efficiency,
            cpu_usage=avg_cpu_usage,
            ram_usage=avg_ram_usage,
            gpu_utilization=avg_gpu_util
        )
    
    def get_detailed_stats(self) -> Dict:
        """获取详细统计信息"""
        metrics = self.get_performance_metrics()
        gpu_stats = self.gpu_manager.get_memory_stats()
        
        stats = {
            "performance_metrics": {
                "avg_fps": metrics.avg_fps,
                "avg_training_time": metrics.avg_training_time,
                "gpu_memory_efficiency": metrics.gpu_memory_efficiency,
                "avg_cpu_usage": metrics.cpu_usage,
                "avg_ram_usage": metrics.ram_usage,
                "avg_gpu_utilization": metrics.gpu_utilization,
            },
            "gpu_stats": gpu_stats,
            "system_info": {
                "cpu_cores": psutil.cpu_count(),
                "total_ram_gb": psutil.virtual_memory().total / 1024**3,
                "available_ram_gb": psutil.virtual_memory().available / 1024**3,
            }
        }
        
        if torch.cuda.is_available():
            stats["cuda_info"] = {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
            }
        
        return stats
    
    def optimize_performance(self) -> Dict[str, bool]:
        """性能优化"""
        optimizations = {}
        
        # GPU内存优化
        gpu_optimized = self.gpu_manager.optimize_memory()
        optimizations["gpu_memory_optimized"] = gpu_optimized
        
        # 检查是否需要降低batch size
        gpu_info = self.gpu_manager.get_memory_info()
        if gpu_info.used / gpu_info.total > 0.9:
            optimizations["suggest_reduce_batch_size"] = True
            logger.warning("建议降低batch_size以减少GPU内存使用")
        else:
            optimizations["suggest_reduce_batch_size"] = False
        
        # 检查CPU性能
        if self.cpu_usage and np.mean(list(self.cpu_usage)[-10:]) > 90:
            optimizations["suggest_reduce_workers"] = True
            logger.warning("建议减少数据加载器的worker数量")
        else:
            optimizations["suggest_reduce_workers"] = False
        
        return optimizations
    
    def save_performance_report(self, filepath: Union[str, Path]):
        """保存性能报告"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.get_detailed_stats()
        
        # 添加时间戳
        stats["timestamp"] = time.time()
        stats["report_generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能报告已保存到: {filepath}")


class WindowsOptimizer:
    """Windows特定优化器"""
    
    def __init__(self):
        self.is_windows = os.name == 'nt'
        if not self.is_windows:
            logger.warning("WindowsOptimizer仅在Windows系统上有效")
    
    def optimize_windows_settings(self) -> Dict[str, bool]:
        """优化Windows设置"""
        if not self.is_windows:
            return {}
        
        optimizations = {}
        
        try:
            # 设置高性能电源计划
            result = subprocess.run([
                'powercfg', '/setactive', 'SCHEME_MIN'
            ], capture_output=True, text=True)
            optimizations["high_performance_power_plan"] = result.returncode == 0
            
            # 禁用Windows Defender实时保护（需要管理员权限）
            # 注意：这可能会影响系统安全，仅建议在训练环境中使用
            
            # 设置进程优先级为高
            current_process = psutil.Process()
            try:
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                optimizations["high_priority_process"] = True
            except:
                optimizations["high_priority_process"] = False
            
            # 设置亲和性（如果是多核CPU）
            cpu_count = psutil.cpu_count()
            if cpu_count > 4:
                # 使用一半的CPU核心以避免过热
                affinity = list(range(0, cpu_count, 2))
                try:
                    current_process.cpu_affinity(affinity)
                    optimizations["cpu_affinity_set"] = True
                except:
                    optimizations["cpu_affinity_set"] = False
            
        except Exception as e:
            logger.error(f"Windows优化失败: {e}")
        
        return optimizations
    
    def check_windows_requirements(self) -> Dict[str, bool]:
        """检查Windows环境要求"""
        checks = {}
        
        if not self.is_windows:
            return checks
        
        try:
            # 检查.NET Framework
            result = subprocess.run([
                'reg', 'query', 
                'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\NET Framework Setup\\NDP\\v4\\Full',
                '/v', 'Release'
            ], capture_output=True, text=True)
            checks["dotnet_framework"] = result.returncode == 0
            
            # 检查Visual C++ Redistributable
            vcredist_paths = [
                "C:\\Windows\\System32\\vcruntime140.dll",
                "C:\\Windows\\System32\\msvcp140.dll",
            ]
            checks["vcredist_installed"] = all(os.path.exists(path) for path in vcredist_paths)
            
            # 检查DirectX
            directx_path = "C:\\Windows\\System32\\d3d11.dll"
            checks["directx_available"] = os.path.exists(directx_path)
            
        except Exception as e:
            logger.error(f"Windows环境检查失败: {e}")
        
        return checks


# 全局性能监控器实例
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def start_performance_monitoring():
    """启动全局性能监控"""
    monitor = get_performance_monitor()
    monitor.start_monitoring()

def stop_performance_monitoring():
    """停止全局性能监控"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()

def optimize_for_training():
    """训练前的性能优化"""
    logger.info("开始训练前性能优化...")
    
    # GPU优化
    gpu_manager = GPUMemoryManager()
    gpu_manager.optimize_memory()
    
    # Windows优化
    windows_optimizer = WindowsOptimizer()
    win_optimizations = windows_optimizer.optimize_windows_settings()
    
    # 性能监控
    start_performance_monitoring()
    
    logger.info("性能优化完成")
    return {
        "gpu_optimized": True,
        "windows_optimizations": win_optimizations,
        "monitoring_started": True,
    }