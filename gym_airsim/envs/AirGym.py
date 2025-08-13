from settings_folder import settings
import msgs
# from gym.envs.classic_control import rendering  # Commented out for gym compatibility
from environment_randomization.game_config_handler_class import *
import gymnasium as gym
import collections
from gymnasium import spaces
from gymnasium.utils import seeding
from gym_airsim.envs.airlearningclient import *
from common.utils import *
import time
import airsim


class AirSimEnv(gym.Env):
    def __init__(self, need_render=False):

        # if need_render is True, then we can use the 2d windows to render the env

        STATE_RGB_H, STATE_RGB_W = 112,112

        self.stack_frames=4

        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.stack_frames, STATE_RGB_H, STATE_RGB_W))


        #eithor your speed is more than 2, or your duration is more than 0.4
        #otherwise, it dose not work well !
        if (settings.control_mode == "moveByVelocity"):
            self.action_space = spaces.Box(np.array([-2.0, -2.0]),
                                           np.array([+2.0, +2.0]),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(8)

        #UE4 env config
        self.game_config_handler = GameConfigHandler()

        #uav api
        self.airgym = AirLearningClient()
        
        # Initialize safe domain randomizer (disabled to prevent crashes)
        # Only use JSON-based randomization for now
        self.use_visual_randomization = False

        #reset the env var
        self.success_count = 0
        self.episodeN = 0
        self.stepN = 0
        # Initialize with default goal, will be updated by randomization
        self.goal = airsimize_coordinates([10.0, -8.25, 0])  # From EnvGenConfig.json


        self.prev_state = self.init_state_f()
        self.init_state = self.prev_state
        self.success = False
        self.level=0
        self.success_deque = collections.deque(maxlen=100)
        
        # 智能碰撞恢复机制变量
        self.collision_count = 0          # 连续碰撞计数
        self.steps_since_collision = 0    # 距离上次碰撞的步数
        self.collision_reset_threshold = 3  # 连续碰撞3次触发重置
        self.collision_reset_interval = 5   # 5步后重置计数
        self.safe_zone_radius = 3.0         # 安全区域搜索半径
        self.start_position = None          # 起点位置，用于重置
        self.prev_distance = None           # 上一步的距离，用于奖励计算
        
        self.seed()

        self.need_render=need_render
        if self.need_render:
            self.viewer = rendering.Viewer(1000, 1000)

    def getGoal(self):
        return self.goal

    def get_space(self):
        return self.observation_space,self.action_space

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def print_msg_of_inspiration(self):
        if (self.success_count %2 == 0):
            print("---------------:) :) :) Success, Be Happy (: (: (:------------ !!!\n")
        elif (self.success_count %3 == 0):
            print("---------------:) :) :) Success, Shake Your Butt (: (: (:------------ !!!\n")
        else:
            print("---------------:) :) :) Success, Oh Yeah! (: (: (:------------ !!!\n")

    def init_state_f(self):
        now = self.airgym.drone_pos()[:2]
        pry = self.airgym.get_ryp()
        d=[]
        for i in range(self.stack_frames):
            d.append(self.airgym.getScreenDepth())
            time.sleep(0.03)
        self.r_yaw = self.airgym.goal_direction(self.goal, now)
        self.relative_position = self.airgym.get_distance(self.goal)
        self.velocity = self.airgym.drone_velocity()
        self.speed = self.velocity[2]
        inform=np.concatenate((self.relative_position, self.velocity, pry, self.r_yaw))
        d=np.stack(d)
        return [d, inform]

    def state(self):
        now = self.airgym.drone_pos()[:2]
        pry = self.airgym.get_ryp()
        self.r_yaw = self.airgym.goal_direction(self.goal, now)
        self.relative_position = self.airgym.get_distance(self.goal)
        self.velocity = self.airgym.drone_velocity()
        self.speed = self.velocity[2]
        inform=np.concatenate((self.relative_position, self.velocity, pry, self.r_yaw))
        return inform

    def computeReward(self, now):
        # 计算到目标的距离
        distance_now = np.sqrt(np.power((self.goal[0] - now[0]), 2)
                               + np.power((self.goal[1] - now[1]), 2)
                               )
        
        # 使用传入的位置计算方向
        now_pos = now[:2]  # 只取X,Y坐标
        r_yaw = self.airgym.goal_direction(self.goal, now_pos)

        # 距离奖励：距离越近奖励越高
        r = -distance_now * 0.1  # 增加距离惩罚权重
        
        # 方向奖励：朝向目标方向有额外奖励
        if math.cos(r_yaw) >= 0:
            r += self.speed * math.cos(r_yaw) * 0.5
            
        # 距离改善奖励
        if hasattr(self, 'prev_distance') and self.prev_distance is not None:
            distance_improvement = self.prev_distance - distance_now
            r += distance_improvement * 2.0  # 距离减少时给正奖励
        
        self.prev_distance = distance_now

        return r


    #根据控制飞机飞的方式不同，动作空间和step会有不同
    #根据不同输入，obs space会有不同
    def step(self, action):

        self.stepN += 1
        
        # Handle action format based on control mode
        if (settings.control_mode == "moveByVelocity"):
            # For continuous control, keep action as is (should be [delta_x, delta_y])
            if hasattr(action, '__len__') and len(action) == 1 and hasattr(action[0], '__len__'):
                # Handle case where action is wrapped: [[delta_x, delta_y]]
                action = action[0]
            # action should now be [delta_x, delta_y] for continuous control
            collided = self.airgym.take_continious_action(action)
        else:
            # For discrete control, extract the action index
            if hasattr(action, '__len__'):
                action = action[0]
            collided = self.airgym.take_discrete_action(action)

        #self.airgym.client.simPause(True)

        #update state
        inform = self.state()

        d=self.airgym.getScreenDepth()
        state = []
        for i in range(self.stack_frames):
            if i <(self.stack_frames-1):
                state.append(self.prev_state[0][i+1])
            else:
                state.append(d)
        state=[state, inform]
        now = self.airgym.drone_pos()

        print("ENter Step" + str(self.stepN))
        print("Relative Position:" + str(self.relative_position))
        print("Success count:",self.success_count)
        #print("Speed:"+str(self.speed))
        print("Action:",action)
        print("Goal:" + str(self.goal))
        print(f"Current UAV position: [{now[0]:.3f}, {now[1]:.3f}, {now[2]:.3f}]")

        distance = np.sqrt(np.power((self.goal[0] - now[0]), 2)
                           +np.power((self.goal[1] - now[1]), 2)
                           )
        
        print(f"🔍 REWARD DEBUG:")
        print(f"   Distance to goal: {distance:.3f} (threshold: {settings.success_distance_to_goal})")
        print(f"   Collision status: {collided}")
        print(f"   Current step: {self.stepN} (max: {settings.nb_max_episodes_steps})")
        print(f"   UAV Z: {now[2]:.3f} (limits: [-10.0, 2.0])")

        if distance < settings.success_distance_to_goal:
            print("✅ SUCCESS - Distance within threshold!")
            self.success_count += 1
            done = True
            self.print_msg_of_inspiration()
            self.success = True
            msgs.success = True
            reward = 20.0

        elif collided == True:
            print("💥 COLLISION detected!")
            
            # 更新碰撞计数和步数统计
            self.collision_count += 1
            self.steps_since_collision = 0
            
            # 渐进惩罚机制
            if self.collision_count == 1:
                reward = -5.0
                print(f"第1次碰撞，轻微惩罚: {reward}")
                done = False
                
            elif self.collision_count == 2:
                reward = -15.0
                print(f"第2次碰撞，中度惩罚: {reward}，强制悬停2秒")
                # 强制悬停2秒
                self.airgym.client.hoverAsync().join()
                time.sleep(2.0)
                done = False
                
            elif self.collision_count >= 3:
                reward = -30.0
                print(f"第3次及以上碰撞，重度惩罚: {reward}，触发智能重置")
                # 执行智能重置
                self.execute_collision_recovery()
                done = False  # 不终止episode，保留训练数据
                
            self.success = False

        elif self.stepN >= settings.nb_max_episodes_steps:
            print("⏰ MAX STEPS reached!")
            done = True
            reward = -20.0
            self.success = False
        elif (now[2] < -10.0) or (now[2] > 2.0):  # Penalize for flying away too high or too low (fixed limits)
            print(f"🚁 HEIGHT VIOLATION! Z={now[2]:.3f} outside [-10.0, 2.0]")
            done = True
            reward = -20.0
            self.success = False

        else:
            # 更新非碰撞步数计数
            self.steps_since_collision += 1
            
            # 如果距离上次碰撞超过间隔步数，重置碰撞计数
            if self.steps_since_collision >= self.collision_reset_interval:
                if self.collision_count > 0:
                    print(f"碰撞计数重置：{self.collision_count} -> 0 (间隔{self.steps_since_collision}步)")
                    self.collision_count = 0
            
            reward = self.computeReward(now)
            print(f"📊 CONTINUING - Computed reward: {reward:.3f}")
            done = False
            self.success = False

        # Todo: penalize for more crazy and unstable actions
        print("rew:", reward)

        self.prev_state = state


        if (done):
            if self.success:
                self.success_deque.append(1)
            else:
                self.success_deque.append(0)
            self.on_episode_end()

        return state, reward, done, None


    def on_episode_end(self):
        # 重置碰撞计数
        self.collision_count = 0
        self.steps_since_collision = 0
        # 重置距离追踪
        self.prev_distance = None
        
    def find_safe_position_near_start(self):
        """在起点附近寻找安全区域"""
        if self.start_position is None:
            # 如果没有起点，使用当前位置
            self.start_position = self.airgym.drone_pos()
            
        max_attempts = 10
        for attempt in range(max_attempts):
            # 在起点周围随机选择位置
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(1.0, self.safe_zone_radius)
            
            safe_x = self.start_position[0] + distance * np.cos(angle)
            safe_y = self.start_position[1] + distance * np.sin(angle)
            safe_z = self.start_position[2]  # 保持相同高度
            
            safe_position = [safe_x, safe_y, safe_z]
            
            # 简单的安全检查：不要太接近边界
            if (-50 < safe_x < 50 and -50 < safe_y < 50 and -10 < safe_z < 2):
                return safe_position
                
        # 如果找不到安全位置，回到起点
        return self.start_position
        
    def execute_collision_recovery(self):
        """执行碰撞恢复：移动到安全区域"""
        print(f"执行碰撞恢复：连续碰撞{self.collision_count}次")
        
        # 寻找安全位置
        safe_position = self.find_safe_position_near_start()
        
        try:
            # 移动到安全位置
            self.airgym.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(safe_position[0], safe_position[1], safe_position[2]),
                    airsim.Quaternionr(0, 0, 0, 1)
                ),
                True  # ignore_collision
            )
            
            # 重置碰撞状态
            time.sleep(0.5)  # 给仿真器时间处理
            
            print(f"成功重置到安全位置: ({safe_position[0]:.2f}, {safe_position[1]:.2f}, {safe_position[2]:.2f})")
            
        except Exception as e:
            print(f"碰撞恢复失败: {e}")
            # 如果重置失败，至少重置计数
            pass
            
        # 重置碰撞计数
        self.collision_count = 0
        self.steps_since_collision = 0


    def on_episode_start(self):
        self.stepN = 0
        self.episodeN += 1

    def reset(self):

        # 完全禁用curriculum learning和随机化用于调试
        print("Curriculum learning and randomization disabled for testing")
        
        if self.need_render:
            self.viewer.geoms.clear()
            self.viewer.onetime_geoms.clear()
        print("enter reset")
        
        # 设置更近的目标，便于学习
        self.goal = utils.airsimize_coordinates([5.0, -4.0, 0])  # 减少到约6.4米距离
        print(f"Fixed goal set to: {self.goal} (closer for better learning)")
        
        self.airgym.unreal_reset()
        print("done unreal_resetting")
        time.sleep(4)# if your pc is a rubbish then make it sleep long
        self.airgym.AirSim_reset()
        print("done arisim reseting")

        self.airgym.client.takeoffAsync().join()
        
        # 清除起飞时的碰撞状态（起飞可能触发误报）
        time.sleep(0.5)  # 等待起飞完成
        try:
            collision_info = self.airgym.client.simGetCollisionInfo()
            if collision_info.has_collided:
                print("清除起飞时的误报碰撞状态")
        except:
            pass

        now = self.airgym.drone_pos()
        
        # 记录起点位置用于碰撞恢复
        if self.start_position is None:
            self.start_position = now.copy()
            print(f"记录起点位置: ({now[0]:.2f}, {now[1]:.2f}, {now[2]:.2f})")

        ##sometimes there may occur something you can't imagine! Just like your uav is dancing~

        # 修复高度检查逻辑：在这个坐标系统中，Z坐标为正数表示高度
        # 检查无人机是否成功起飞到合理高度（至少0.5米）
        max_retries = 3  # 限制重试次数避免无限循环
        retry_count = 0
        
        while now[2] < 0.5 and retry_count < max_retries:
            retry_count += 1
            print(f"UAV position issue (height: {now[2]:.2f}m), restarting without randomization (attempt {retry_count}/{max_retries})")
            
            self.goal = utils.airsimize_coordinates([5.0, -4.0, 0])  # 更近的固定目标
            self.airgym.unreal_reset()
            print("done unreal_resetting")
            time.sleep(4)
            self.airgym.AirSim_reset()
            self.airgym.client.takeoffAsync().join()
            now = self.airgym.drone_pos()
        
        if retry_count >= max_retries:
            print(f"警告：无人机起飞重试达到最大次数，当前高度: {now[2]:.2f}m，继续执行...")

        self.airgym.client.moveByVelocityZAsync(0,0,self.airgym.z, 1).join()

        # 保持固定目标
        print(f"Final goal confirmed: {self.goal}")

        self.on_episode_start()
        print("done on episode start")

        state = self.init_state_f()
        self.prev_state = state

        return state

    def randomize_env(self):
        # 完全禁用环境随机化用于模型测试
        print("Environment randomization disabled for testing")
        
        # 只更新目标位置从配置文件
        try:
            current_goal = self.game_config_handler.get_cur_item("End")
            if isinstance(current_goal, list) and len(current_goal) >= 2:
                self.goal = utils.airsimize_coordinates(current_goal)
                print(f"Goal loaded from config: {self.goal}")
            else:
                # 使用固定目标用于测试
                self.goal = utils.airsimize_coordinates([10.0, -8.25, 0])
                print(f"Using fixed goal for testing: {self.goal}")
        except Exception as e:
            print(f"Error reading goal, using default: {e}")
            self.goal = utils.airsimize_coordinates([10.0, -8.25, 0])


    def updateJson(self, *args):
        self.game_config_handler.update_json(*args)
        print(f"JSON updated with args: {args}")

    def getItemCurGameConfig(self, key):
        return self.game_config_handler.get_cur_item(key)

    def setRangeGameConfig(self, *args):
        self.game_config_handler.set_range(*args)
        print(f"Range config set with args: {args}")

    def getRangeGameConfig(self, key):
        return self.game_config_handler.get_range(key)

    def sampleGameConfig(self, *arg):
        self.game_config_handler.sample(*arg, np_random=self.np_random)
        print(f"Game config sampled for: {arg}")
        # Update JSON configuration file
        self.updateJson(*arg)

    def render(self, mode='human', close=False):
        # 画一个直径为 30 的circle
        #self.viewer.geoms.clear()
        #self.viewer.onetime_geoms.clear()

        goal = rendering.make_circle(30)
        goal.set_color(0.25, 0.78, 0.1)
        # 添加一个平移操作
        circle_transform = rendering.Transform(translation=(self.goal[0]*10+500, self.goal[1]*10+500))
        # 让圆添加平移这个属性
        goal.add_attr(circle_transform)
        self.viewer.add_geom(goal)

        uav = rendering.make_circle(10)
        uav.set_color(0.78, 0.37, 0.66)
        now = self.airgym.drone_pos()
        uav_transform = rendering.Transform(translation=(now[0]*10 + 500, now[1]*10 + +500))
        uav.add_attr(uav_transform)
        self.viewer.add_geom(uav)

        size = self.game_config_handler.get_cur_item("ArenaSize")
        h = size[0] * 10
        w = size[1] * 10

        line1 = rendering.Line(((1000 - h) / 2, (1000 - w) / 2), ((h + 1000) / 2, (1000 - w) / 2))
        line2 = rendering.Line(((1000 - h) / 2, (1000 - w) / 2), ((1000 - h) / 2, (w + 1000) / 2))
        line3 = rendering.Line(((h + 1000) / 2, (1000 - w) / 2), ((h + 1000) / 2, (1000 + w) / 2))
        line4 = rendering.Line(((1000 - h) / 2, (1000 + w) / 2), ((h + 1000) / 2, (1000 + w) / 2))
        # 给元素添加颜色
        line1.set_color(0, 0, 0)
        line2.set_color(0, 0, 0)
        line3.set_color(0, 0, 0)
        line4.set_color(0, 0, 0)
        self.viewer.add_geom(line1)
        self.viewer.add_geom(line2)
        self.viewer.add_geom(line3)
        self.viewer.add_geom(line4)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')