#!/usr/bin/env python3
"""
启动AirSim仿真环境的便捷脚本
支持多种预编译场景
"""

import os
import sys
import subprocess
import argparse
import signal
import time
from pathlib import Path


class SimulationLauncher:
    """仿真环境启动器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent / "AirSim_Precompiled"
        self.current_process = None
        
        # 可用的仿真场景
        self.scenarios = {
            'blocks': {
                'path': self.base_dir / "LinuxBlocks1.8.1" / "LinuxNoEditor" / "Blocks.sh",
                'description': "经典Blocks环境 - 适合无人机导航训练"
            },
            'maze': {
                'path': self.base_dir / "SimpleMaze" / "Car_Maze.sh", 
                'description': "简单迷宫环境 - 适合路径规划测试"
            }
        }
    
    def list_scenarios(self):
        """列出所有可用场景"""
        print("可用的仿真场景:")
        print("-" * 50)
        for name, info in self.scenarios.items():
            status = "✓" if info['path'].exists() else "✗"
            print(f"{status} {name:10s} - {info['description']}")
        print()
    
    def check_requirements(self):
        """检查环境要求"""
        print("检查仿真环境要求...")
        
        # 检查AirSim Python API
        try:
            import airsim
            print("✓ AirSim Python API 已安装")
        except ImportError:
            print("✗ AirSim Python API 未安装")
            print("  请运行: pip install airsim")
            return False
        
        # 检查预编译场景
        missing_scenarios = []
        for name, info in self.scenarios.items():
            if not info['path'].exists():
                missing_scenarios.append(name)
        
        if missing_scenarios:
            print(f"✗ 缺少场景: {', '.join(missing_scenarios)}")
            return False
        else:
            print("✓ 所有预编译场景已就绪")
        
        return True
    
    def make_executable(self, script_path):
        """确保脚本可执行"""
        if script_path.exists():
            os.chmod(script_path, 0o755)
            return True
        return False
    
    def start_scenario(self, scenario_name, headless=False):
        """启动指定场景"""
        if scenario_name not in self.scenarios:
            print(f"错误: 未知场景 '{scenario_name}'")
            self.list_scenarios()
            return False
        
        scenario = self.scenarios[scenario_name]
        script_path = scenario['path']
        
        if not script_path.exists():
            print(f"错误: 场景文件不存在: {script_path}")
            return False
        
        # 确保脚本可执行
        self.make_executable(script_path)
        
        print(f"启动场景: {scenario['description']}")
        print(f"执行文件: {script_path}")
        
        try:
            # 构建启动命令
            cmd = [str(script_path)]
            if headless:
                cmd.extend(['-RenderOffscreen', '-windowed', '-ResX=640', '-ResY=480'])
            
            # 启动仿真环境
            self.current_process = subprocess.Popen(
                cmd,
                cwd=script_path.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            print(f"仿真环境已启动 (PID: {self.current_process.pid})")
            print("等待环境初始化...")
            
            # 等待几秒让环境初始化
            time.sleep(3)
            
            # 检查进程是否还在运行
            if self.current_process.poll() is None:
                print("✓ 仿真环境启动成功!")
                print("按 Ctrl+C 停止仿真")
                return True
            else:
                stdout, stderr = self.current_process.communicate()
                print("✗ 仿真环境启动失败")
                if stderr:
                    print(f"错误信息: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"启动失败: {e}")
            return False
    
    def stop_scenario(self):
        """停止当前运行的场景"""
        if self.current_process and self.current_process.poll() is None:
            print("正在停止仿真环境...")
            self.current_process.terminate()
            
            # 等待进程结束
            try:
                self.current_process.wait(timeout=5)
                print("✓ 仿真环境已停止")
            except subprocess.TimeoutExpired:
                print("强制停止仿真环境...")
                self.current_process.kill()
                self.current_process.wait()
                print("✓ 仿真环境已强制停止")
    
    def setup_signal_handler(self):
        """设置信号处理器"""
        def signal_handler(sig, frame):
            print("\n收到停止信号...")
            self.stop_scenario()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run_interactive(self):
        """交互式运行"""
        print("AirSim 仿真环境启动器")
        print("=" * 30)
        
        self.list_scenarios()
        
        while True:
            try:
                choice = input("选择场景 (blocks/maze) 或输入 'q' 退出: ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice in self.scenarios:
                    headless = input("无头模式运行? (y/N): ").strip().lower() == 'y'
                    
                    if self.start_scenario(choice, headless):
                        try:
                            # 保持运行直到用户中断
                            self.current_process.wait()
                        except KeyboardInterrupt:
                            pass
                        finally:
                            self.stop_scenario()
                    break
                else:
                    print("无效选择，请重试")
                    
            except KeyboardInterrupt:
                print("\n退出...")
                break


def main():
    parser = argparse.ArgumentParser(description="AirSim 仿真环境启动器")
    parser.add_argument('--scenario', choices=['blocks', 'maze'], 
                       help='要启动的场景')
    parser.add_argument('--headless', action='store_true',
                       help='无头模式运行')
    parser.add_argument('--list', action='store_true',
                       help='列出可用场景')
    parser.add_argument('--check', action='store_true',
                       help='检查环境要求')
    
    args = parser.parse_args()
    
    launcher = SimulationLauncher()
    launcher.setup_signal_handler()
    
    if args.list:
        launcher.list_scenarios()
    elif args.check:
        if launcher.check_requirements():
            print("✓ 环境检查通过")
        else:
            print("✗ 环境检查失败")
            sys.exit(1)
    elif args.scenario:
        if launcher.check_requirements():
            launcher.start_scenario(args.scenario, args.headless)
            if launcher.current_process:
                try:
                    launcher.current_process.wait()
                except KeyboardInterrupt:
                    pass
                finally:
                    launcher.stop_scenario()
        else:
            sys.exit(1)
    else:
        launcher.run_interactive()


if __name__ == "__main__":
    main()