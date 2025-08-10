#!/usr/bin/env python
"""
快速测试荧光轨迹效果
使用前请确保AirSim环境已启动
"""

from fluorescent_trail import FluorescentTrail
import time

def main():
    print("荧光轨迹测试程序")
    print("请确保AirSim环境已启动...")
    
    trail = FluorescentTrail()
    
    if not trail.connect():
        print("无法连接到AirSim，请检查环境是否启动")
        return
    
    print("\n开始荧光轨迹展示...")
    print("在AirSim窗口中按 'T' 键显示轨迹")
    
    # 展示不同的荧光效果
    effects = [
        ("超细轨迹", "ultra_thin"),
        ("激光红", "laser_red"),
        ("幽灵白", "ghost_white"),
        ("电子蓝", "electric_blue"),
        ("毒液绿", "toxic_green"), 
        ("热粉色", "hot_pink"),
        ("赛博紫", "cyber_purple"),
        ("霓虹橙", "neon_orange"),
        ("等离子体", "plasma")
    ]
    
    for name, effect in effects:
        print(f"\n当前效果: {name}")
        trail.set_neon_trail(effect)
        print("按 Enter 继续到下一个效果...")
        input()
    
    print("\n开始动态变化效果 (10秒)...")
    trail.dynamic_trail_effect(10)
    
    print("\n测试完成！")
    print("现在可以启动训练，轨迹会自动应用荧光效果")

if __name__ == "__main__":
    main()