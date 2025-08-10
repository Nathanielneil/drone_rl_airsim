#!/usr/bin/env python
"""
荧光轨迹设置脚本
为无人机训练添加超酷的荧光轨迹效果
"""

import airsim
import time
import colorsys

class FluorescentTrail:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.current_effect = 0
        
    def connect(self):
        """连接到AirSim"""
        try:
            self.client.confirmConnection()
            print("连接到AirSim成功！")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def set_neon_trail(self, color_type="rainbow"):
        """设置霓虹灯效果轨迹"""
        effects = {
            "rainbow": self._rainbow_trail,
            "electric_blue": self._electric_blue_trail,
            "toxic_green": self._toxic_green_trail,
            "hot_pink": self._hot_pink_trail,
            "cyber_purple": self._cyber_purple_trail,
            "plasma": self._plasma_trail
        }
        
        if color_type in effects:
            effects[color_type]()
        else:
            print(f"未知效果类型: {color_type}")
    
    def _rainbow_trail(self):
        """彩虹荧光轨迹"""
        print("设置彩虹荧光轨迹...")
        colors = [
            [1.0, 0.0, 0.5, 1.0],  # 荧光红
            [1.0, 0.5, 0.0, 1.0],  # 荧光橙
            [1.0, 1.0, 0.0, 1.0],  # 荧光黄
            [0.0, 1.0, 0.0, 1.0],  # 荧光绿
            [0.0, 1.0, 1.0, 1.0],  # 荧光青
            [0.0, 0.0, 1.0, 1.0],  # 荧光蓝
            [1.0, 0.0, 1.0, 1.0],  # 荧光紫
        ]
        
        for i, color in enumerate(colors):
            self.client.simSetTraceLine(color, 15 + i * 5)
            time.sleep(0.5)
    
    def _electric_blue_trail(self):
        """电子蓝荧光轨迹"""
        print("设置电子蓝荧光轨迹...")
        # 电子蓝：强烈的蓝色带白色光晕
        self.client.simSetTraceLine([0.0, 0.8, 1.0, 1.0], 25)
        time.sleep(0.2)
        self.client.simSetTraceLine([0.3, 0.9, 1.0, 0.8], 30)
    
    def _toxic_green_trail(self):
        """毒液绿荧光轨迹"""
        print("设置毒液绿荧光轨迹...")
        # 毒绿色：明亮的绿色
        self.client.simSetTraceLine([0.0, 1.0, 0.2, 1.0], 20)
        time.sleep(0.2)
        self.client.simSetTraceLine([0.2, 1.0, 0.0, 0.9], 25)
    
    def _hot_pink_trail(self):
        """热粉荧光轨迹"""
        print("设置热粉荧光轨迹...")
        # 热粉色：明亮的粉红
        self.client.simSetTraceLine([1.0, 0.0, 0.7, 1.0], 20)
        time.sleep(0.2)
        self.client.simSetTraceLine([1.0, 0.3, 0.8, 0.8], 25)
    
    def _cyber_purple_trail(self):
        """赛博紫荧光轨迹"""
        print("设置赛博紫荧光轨迹...")
        # 赛博紫：深紫色带电子效果
        self.client.simSetTraceLine([0.6, 0.0, 1.0, 1.0], 20)
        time.sleep(0.2)
        self.client.simSetTraceLine([0.8, 0.3, 1.0, 0.8], 25)
    
    def _plasma_trail(self):
        """等离子体荧光轨迹"""
        print("设置等离子体荧光轨迹...")
        # 等离子体：白色到蓝色渐变
        self.client.simSetTraceLine([1.0, 1.0, 1.0, 1.0], 15)
        time.sleep(0.3)
        self.client.simSetTraceLine([0.7, 0.9, 1.0, 0.9], 20)
        time.sleep(0.3)
        self.client.simSetTraceLine([0.3, 0.7, 1.0, 0.8], 25)
    
    def dynamic_trail_effect(self, duration=10):
        """动态变化的荧光轨迹效果"""
        print(f"启动动态荧光轨迹效果 ({duration}秒)...")
        
        start_time = time.time()
        effects = ["electric_blue", "toxic_green", "hot_pink", "cyber_purple", "plasma"]
        
        while time.time() - start_time < duration:
            effect = effects[int(time.time()) % len(effects)]
            self.set_neon_trail(effect)
            time.sleep(2)
    
    def set_custom_fluorescent(self, r, g, b, thickness=20):
        """设置自定义荧光颜色"""
        # 增强饱和度和亮度以获得荧光效果
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = min(1.0, s * 1.5)  # 增加饱和度
        v = min(1.0, v * 1.3)  # 增加亮度
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        self.client.simSetTraceLine([r, g, b, 1.0], thickness)
        print(f"设置自定义荧光轨迹: RGB({r:.2f}, {g:.2f}, {b:.2f})")

def apply_fluorescent_trail_to_training():
    """在训练过程中应用荧光轨迹"""
    trail = FluorescentTrail()
    if trail.connect():
        # 设置默认的超酷荧光轨迹
        trail.set_neon_trail("electric_blue")
        print("荧光轨迹已激活！按 'T' 键在仿真中查看轨迹")
        return trail
    return None

if __name__ == "__main__":
    # 测试脚本
    trail = FluorescentTrail()
    if trail.connect():
        print("\n荧光轨迹测试开始！")
        print("可用效果: rainbow, electric_blue, toxic_green, hot_pink, cyber_purple, plasma")
        
        # 测试各种效果
        effects = ["electric_blue", "toxic_green", "hot_pink", "cyber_purple", "plasma", "rainbow"]
        
        for effect in effects:
            print(f"\n测试效果: {effect}")
            trail.set_neon_trail(effect)
            time.sleep(3)
        
        print("\n测试完成！现在启动训练并按 'T' 键查看轨迹效果")