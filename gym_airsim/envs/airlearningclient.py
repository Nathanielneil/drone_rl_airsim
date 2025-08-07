import airsim
import numpy as np
import math
import time
import cv2
from settings_folder import settings



class AirLearningClient(object):
    def __init__(self):

        self.last_img = np.zeros((1, 112, 112))
        self.last_grey = np.zeros((112, 112))
        self.last_rgb = np.zeros((112, 112, 3))
        self.width, self.height=84,84 ##deepmind settings

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient(settings.ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        #self.z=-3
        self.z = -0.9

    def goal_direction(self, goal, pos):

        orientation = self.client.simGetVehiclePose().orientation
        pitch, roll, yaw = airsim.to_eularian_angles(orientation)
        yaw = math.degrees(yaw)

        pos_angle = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)
        #return ((math.degrees(track) - 180) % 360) - 180

        return np.array([track])

    def getScreenRGB(self):
        responses = self.client.simGetImage("3d", airsim.ImageType.Scene)
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if ((responses[0].width != 0 or responses[0].height != 0)):
            img_rgba = img1d.reshape((response.height, response.width, 4))
            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            self.last_rgb=rgb
        else:
            print("Something bad happened! Restting AirSim!")
            #self.AirSim_reset()

            rgb=self.last_rgb
        #rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return rgb

    def getScreenDepth(self):
        try:
            # Try different camera names for different environments
            camera_names = ["0", "front", "front_center", ""]
            responses = None
            
            for camera_name in camera_names:
                try:
                    responses = self.client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, True, False)])
                    if responses and len(responses) > 0 and responses[0].width > 0:
                        break
                except:
                    continue
            
            if not responses:
                # If all camera requests fail, try without specifying camera
                responses = self.client.simGetImages([airsim.ImageRequest("", airsim.ImageType.DepthVis, True, False)])
                
        except Exception as e:
            print(f"Error getting depth image: {e}")
            # Return a default depth image if all attempts fail
            return np.zeros((112, 112))
        #responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis,True, False)])

        if (responses == None):
            print("Camera is not returning image!")
            print("Image size:" + str(responses[0].height) + "," + str(responses[0].width))
            img = [np.array([0]) for _ in responses]
        else:
            img = []
            for res in responses:
                img.append(np.array(res.image_data_float, dtype=np.float))
            img = np.stack(img, axis=0)


        ##pre-process for depth img
        img = img.clip(max=20)
        scale=255/20
        img=img*scale#, dtype=np.uint8)

        img2d=[]
        for i in range(len(responses)):
            if ((responses[i].width != 0 or responses[i].height != 0)):
                reshaped_img = np.reshape(img[i], (responses[i].height, responses[i].width))
                # Resize to 112x112 to match observation space
                import cv2
                resized_img = cv2.resize(reshaped_img, (112, 112), interpolation=cv2.INTER_AREA)
                img2d.append(resized_img)
            else:
                print("Something bad happened! Restting AirSim!")
                if hasattr(self, 'last_img') and len(self.last_img) > i:
                    img2d.append(self.last_img[i])
                else:
                    # Create a default 112x112 image if no last image
                    img2d.append(np.zeros((112, 112)))

        self.last_img = np.stack(img2d, axis=0)

        if len(img2d)>1:
            return img2d
        else:
            return img2d[0]

    def get_ryp(self):
        orientation = self.client.simGetVehiclePose().orientation
        pitch, roll, yaw = airsim.to_eularian_angles(orientation)
        return np.array([pitch, roll, yaw])  # Return all 3 angles as originally intended

    def drone_pos(self):
        pos = self.client.simGetVehiclePose().position
        x = pos.x_val
        y = pos.y_val
        z = pos.z_val

        return np.array([x, y, z])

    def drone_velocity(self):
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        v_x = vel.x_val
        v_y = vel.y_val
        v_z = vel.z_val
        speed = np.sqrt(v_x ** 2 + v_y ** 2)
        return np.array([v_x, v_y, speed])

    def get_distance(self, goal):
        now = self.client.simGetVehiclePose().position
        xdistance = (goal[0] - now.x_val)
        ydistance = (goal[1] - now.y_val)
        #zdistance = (goal[2] - now.z_val)
        euclidean = np.sqrt(np.power(xdistance,2) + np.power(ydistance,2))
        return np.array([xdistance, ydistance])

    def get_velocity(self):
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])

    def AirSim_reset(self):
        self.client=airsim.MultirotorClient(settings.ip)
        connection_established = False
        max_retries = 5
        retry_count = 0
        
        # wait till connected to the multi rotor
        while not connection_established and retry_count < max_retries:
            try:
                time.sleep(1)
                self.client.confirmConnection()  # This will throw exception if not connected
                connection_established = True
                print("Connection established successfully!")
            except Exception as e:
                retry_count += 1
                print(f"Connection attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    time.sleep(2)
                    self.client = airsim.MultirotorClient(settings.ip)
                else:
                    print("Max retries reached, using existing connection")
                    connection_established = True

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def unreal_reset(self):
        #!!!这个时间间隔很重要！！！！
        try:
            print("🔄 Triggering Unreal environment regeneration...")
            self.client.simPause(False)  # Ensure simulation is running
            
            # This triggers UE4 to read EnvGenConfig.json and regenerate environment
            # The time intervals are critical for proper environment generation
            result = self.client.reset()  # Use basic reset first
            time.sleep(1.5)  # Wait for reset to process
            
            # Try Unreal-specific reset if available
            try:
                self.client.resetUnreal(1.5, 2.5)  # Original parameters from AirLearning
                print("✅ Unreal environment regenerated successfully")
            except Exception as e:
                # Fallback to basic reset if resetUnreal is not available
                print(f"⚠️  resetUnreal not available, using basic reset: {e}")
                self.client.reset()
                
        except Exception as e:
            print(f"⚠️  Unreal reset failed: {e}")
            print("Continuing with basic AirSim reset")


    def take_continious_action(self, action):

        if(settings.control_mode=="moveByVelocity"):
            action=np.clip(action, -0.3, 0.3)

            detla_x = action[0]
            detla_y = action[1]
            v=self.drone_velocity()
            v_x = v[0] + detla_x
            v_y = v[1] + detla_y

            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, 0.35, 1, yaw_mode).join()

        else:
            raise NotImplementedError

        # Use new collision detection API
        collision_info = self.client.simGetCollisionInfo()
        collided = collision_info.has_collided

        return collided
        #Todo : Stabilize drone


    def straight(self, speed, duration):
        orientation = self.client.simGetVehiclePose().orientation
        pitch, roll, yaw = airsim.to_eularian_angles(orientation)
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        self.client.moveByVelocityZAsync(vx, vy, self.z, duration, 1, yaw_mode).join()


    def move_right(self, speed, duration):
        orientation = self.client.simGetVehiclePose().orientation
        pitch, roll, yaw = airsim.to_eularian_angles(orientation)
        vx = math.sin(yaw) * speed
        vy = math.cos(yaw) * speed
        self.client.moveByVelocityZAsync(vx, vy, self.z, duration, 0).join()
        start = time.time()
        return start, duration

    def yaw_right(self, rate, duration):
        self.client.rotateByYawRateAsync(rate, duration).join()
        start = time.time()
        return start, duration

    def pitch_up(self, duration):
        self.client.moveByVelocityAsync(0,0,1,duration,1).join()
        start = time.time()
        return start, duration

    def pitch_down(self, duration):
        #yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        self.client.moveByVelocityAsync(0,0,-1,duration,1).join()
        start = time.time()
        return start, duration

    def move_forward_Speed(self, speed_x = 0.5, speed_y = 0.5, duration = 0.5):
        #speedx is in the FLU
        #z = self.drone_pos()[2]
        orientation = self.client.simGetVehiclePose().orientation
        pitch, roll, yaw = airsim.to_eularian_angles(orientation)
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        vx = math.cos(yaw) * speed_x + math.sin(yaw) * speed_y
        vy = math.sin(yaw) * speed_x - math.cos(yaw) * speed_y

        drivetrain = 1
        yaw_mode = airsim.YawMode(is_rate= False, yaw_or_rate = 0)

        self.client.moveByVelocityZAsync(vx = (vx +vel.x_val)/2 ,
                             vy = (vy +vel.y_val)/2 , #do this to try and smooth the movement
                             z = self.z,
                             duration = duration,
                             drivetrain = drivetrain,
                             yaw_mode=yaw_mode
                            ).join()
        start = time.time()
        return start, duration

    def take_discrete_action(self, action):

        if action == 0:
            self.straight(settings.mv_fw_spd_2, settings.rot_dur)
        if action == 1:
            self.straight(settings.mv_fw_spd_3, settings.rot_dur)
        if action == 2:
            #self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur/2)
            #self.straight(settings.mv_fw_spd_3, settings.rot_dur/2)
            self.move_forward_Speed(settings.mv_fw_spd_2*math.cos(0.314),
                                    settings.mv_fw_spd_2*math.sin(0.314), settings.rot_dur)

        if action == 3:
            #self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur / 2)
            #self.straight(settings.mv_fw_spd_4, settings.rot_dur / 2)
            self.move_forward_Speed(settings.mv_fw_spd_3 * math.cos(0.314),
                                    settings.mv_fw_spd_3 * math.sin(0.314), settings.rot_dur)

        if action == 4:
            #self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur / 2)
            #self.straight(settings.mv_fw_spd_4, settings.rot_dur / 2)
            self.move_forward_Speed(settings.mv_fw_spd_2 * math.cos(0.314),
                                    -settings.mv_fw_spd_2 * math.sin(0.314), settings.rot_dur)
        if action == 5:
            #self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur / 2)
            #self.straight(settings.mv_fw_spd_4, settings.rot_dur / 2)
            self.move_forward_Speed(settings.mv_fw_spd_3 * math.cos(0.314),
                                    -settings.mv_fw_spd_3 * math.sin(0.314), settings.rot_dur)

        if action == 6:
            self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur )
        if action == 7:
            self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur)
        '''
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        if action == 0:
            v = self.drone_velocity()
            v_x = v[0] + 0.25
            v_y = v[1] + 0
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()

        if action == 1:
            v = self.drone_velocity()
            v_x = v[0] - 0.25
            v_y = v[1] + 0
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()

        if action == 2:
            v = self.drone_velocity()
            v_x = v[0] + 0
            v_y = v[1] + 0.25
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()

        if action == 3:
            v = self.drone_velocity()
            v_x = v[0] + 0
            v_y = v[1] - 0.25
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()


        
        if action == 0:
            start, duration = self.straight(settings.mv_fw_spd_4, settings.mv_fw_dur)
        if action == 1:
            start, duration = self.straight(settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 2:
            start, duration = self.straight(settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 3:
            start, duration = self.straight(settings.mv_fw_spd_1, settings.mv_fw_dur)

        if action == 4:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_3, settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 5:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_2, settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 6:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_1, settings.mv_fw_spd_1, settings.mv_fw_dur)
        if action == 7:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_3, -settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 8:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_2, -settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 9:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_1, -settings.mv_fw_spd_1, settings.mv_fw_dur)

        if action == 10:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_4, settings.mv_fw_dur)
        if action == 11:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 12:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 13:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_1, settings.mv_fw_dur)

        if action == 14:
            start, duration = self.yaw_right(settings.yaw_rate_1_1, settings.rot_dur)
        if action == 15:
            start, duration = self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur)
        if action == 16:
            start, duration = self.yaw_right(settings.yaw_rate_1_4, settings.rot_dur)
        if action == 17:
            start, duration = self.yaw_right(settings.yaw_rate_1_8, settings.rot_dur)

        if action == 18:
            start, duration = self.yaw_right(settings.yaw_rate_2_1, settings.rot_dur)
        if action == 19:
            start, duration = self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur)
        if action == 20:
            start, duration = self.yaw_right(settings.yaw_rate_2_4, settings.rot_dur)
        if action == 21:
            start, duration = self.yaw_right(settings.yaw_rate_2_8, settings.rot_dur)
        '''

        # Use new collision detection API
        collision_info = self.client.simGetCollisionInfo()
        collided = collision_info.has_collided

        return collided

