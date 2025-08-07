#!/usr/bin/env python3
"""
Implement visual domain randomization for LinuxBlocks1.8.1 using AirSim dynamic object spawning
"""

import airsim
import numpy as np
import random
import time
import json

class VisualDomainRandomizer:
    def __init__(self, client):
        self.client = client
        self.spawned_objects = []
        self.current_level = 0  # 0=easy, 1=medium, 2=hard
        
    def clear_spawned_objects(self):
        """Remove all previously spawned objects"""
        for obj_name in self.spawned_objects:
            try:
                self.client.simDestroyObject(obj_name)
                print(f"Removed object: {obj_name}")
            except:
                pass
        self.spawned_objects = []
    
    def spawn_random_obstacles(self, difficulty_level=0):
        """Spawn random obstacles based on difficulty level"""
        
        # Define obstacle counts per difficulty
        obstacle_counts = {
            0: (5, 12),   # Easy: 5-12 obstacles  
            1: (15, 22),  # Medium: 15-22 obstacles
            2: (25, 35)   # Hard: 25-35 obstacles
        }
        
        arena_sizes = {
            0: 30,  # Easy: 30x30
            1: 45,  # Medium: 45x45  
            2: 60   # Hard: 60x60
        }
        
        min_obs, max_obs = obstacle_counts[difficulty_level]
        arena_size = arena_sizes[difficulty_level]
        num_obstacles = random.randint(min_obs, max_obs)
        
        print(f"Spawning {num_obstacles} obstacles for difficulty level {difficulty_level}")
        print(f"Arena size: {arena_size}x{arena_size}")
        
        # Available object types for obstacles
        obstacle_types = [
            "1m_cube",
            "2m_cube", 
            "Cylinder",
            "Sphere"
        ]
        
        successful_spawns = 0
        
        for i in range(num_obstacles):
            try:
                # Random position within arena
                x = random.uniform(-arena_size/2, arena_size/2)
                y = random.uniform(-arena_size/2, arena_size/2)
                z = random.uniform(-8, -1)  # Height variation
                
                # Random scale
                scale_factor = random.uniform(0.8, 2.5)
                scale = airsim.Vector3r(scale_factor, scale_factor, scale_factor)
                
                # Random rotation
                yaw = random.uniform(0, 360)
                orientation = airsim.to_quaternion(0, 0, np.radians(yaw))
                
                # Create pose
                position = airsim.Vector3r(x, y, z)
                pose = airsim.Pose(position, orientation)
                
                # Choose random obstacle type
                obstacle_type = random.choice(obstacle_types)
                object_name = f"dynamic_obstacle_{i}_{int(time.time())}"
                
                # Attempt to spawn
                result = self.client.simSpawnObject(
                    object_name=object_name,
                    asset_name=obstacle_type,
                    pose=pose,
                    scale=scale,
                    physics_enabled=True
                )
                
                if result:
                    self.spawned_objects.append(result)
                    successful_spawns += 1
                    
                    # Randomize material color
                    self.randomize_object_material(result)
                    
            except Exception as e:
                print(f"Failed to spawn obstacle {i}: {e}")
        
        print(f"Successfully spawned {successful_spawns} obstacles")
        return successful_spawns
    
    def randomize_object_material(self, object_name):
        """Randomize object material/color"""
        try:
            # Random RGB values
            r = random.uniform(0, 1)
            g = random.uniform(0, 1) 
            b = random.uniform(0, 1)
            
            # This API might not be available in all AirSim versions
            # self.client.simSetObjectMaterial(object_name, f"M_RandomColor_{r}_{g}_{b}")
            
        except Exception as e:
            pass  # Material changes not critical
    
    def randomize_lighting(self):
        """Randomize lighting conditions"""
        try:
            # Random time of day
            hour = random.randint(6, 18)
            minute = random.randint(0, 59)
            time_str = f"2023-01-01 {hour:02d}:{minute:02d}:00"
            
            self.client.simSetTimeOfDay(True, time_str)
            print(f"Set time to: {time_str}")
            
        except Exception as e:
            print(f"Lighting randomization failed: {e}")
    
    def randomize_weather(self):
        """Randomize weather conditions"""
        try:
            self.client.simEnableWeather(True)
            
            # Random rain intensity
            rain_intensity = random.uniform(0, 0.5)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, rain_intensity)
            
            # Random wind
            wind_strength = random.uniform(0, 0.3)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Dust, wind_strength)
            
            print(f"Weather: Rain={rain_intensity:.2f}, Wind={wind_strength:.2f}")
            
        except Exception as e:
            print(f"Weather randomization failed: {e}")
    
    def randomize_goal_position(self, arena_size=40):
        """Generate new random goal position"""
        goal_x = random.uniform(-arena_size/3, arena_size/3)
        goal_y = random.uniform(-arena_size/3, arena_size/3)
        goal_z = random.uniform(-3, -1)
        
        goal_position = [goal_x, goal_y, goal_z]
        print(f"New goal position: {goal_position}")
        return goal_position
    
    def apply_full_domain_randomization(self, difficulty_level=0):
        """Apply complete domain randomization"""
        print(f"\n🎲 Applying domain randomization (Difficulty: {difficulty_level})")
        print("="*50)
        
        # Step 1: Clear previous obstacles
        self.clear_spawned_objects()
        time.sleep(0.5)
        
        # Step 2: Spawn new obstacles
        num_spawned = self.spawn_random_obstacles(difficulty_level)
        
        # Step 3: Randomize environmental conditions
        self.randomize_lighting()
        self.randomize_weather()
        
        # Step 4: Generate new goal
        new_goal = self.randomize_goal_position()
        
        print(f"✅ Domain randomization complete!")
        print(f"   - Obstacles: {num_spawned}")
        print(f"   - New goal: {new_goal}")
        print("="*50)
        
        return new_goal

def test_visual_domain_randomization():
    """Test the visual domain randomization system"""
    
    print("🎮 Testing Visual Domain Randomization with LinuxBlocks1.8.1")
    print("Make sure AirSim is running!")
    
    # Connect to AirSim
    client = airsim.MultirotorClient()
    try:
        client.confirmConnection()
        print("✅ Connected to AirSim")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    # Initialize randomizer
    randomizer = VisualDomainRandomizer(client)
    
    # Enable API control
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Test each difficulty level
    for difficulty in [0, 1, 2]:
        difficulty_names = ["Easy", "Medium", "Hard"]
        print(f"\n🧪 Testing {difficulty_names[difficulty]} difficulty...")
        
        goal = randomizer.apply_full_domain_randomization(difficulty)
        
        input(f"Press Enter to continue to next difficulty level...")
    
    print("\n✅ Visual domain randomization test complete!")
    print("You should see obstacles appearing and disappearing in the simulation!")
    
    return True

if __name__ == "__main__":
    test_visual_domain_randomization()