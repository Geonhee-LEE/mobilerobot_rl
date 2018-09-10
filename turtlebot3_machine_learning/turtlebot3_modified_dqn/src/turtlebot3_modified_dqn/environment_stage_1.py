#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

    def getGoalDistace(self):
        '''
        Calculate the range between robot and goal and Return the range.

        Parameters:
        - 

        Return:
        - goal distance from robot's position to the Goal position.
        '''

        # hypot(x,y): Return the Euclidean norm, sqrt(x*x + y*y)
        # round(x[, n]): Return x rounded to n digits from the decimal point
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        '''
        Calculate heading value which yaw subtract from goal_angle based on Odometry message achieved from /Odom topic

        Parameters:
        - odom: Odometry topic message, which the topic name is '/odom'
        
        Return:
        - 
        '''
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        # heading variable mean the difference between robot's orientation and goal
        self.heading = round(heading, 2)

    def getState(self, scan):
        '''
        Process laser data to be satisfied with constraint and integrate laser data and heading and distance variables into one array.
        
        Parameters:
        - scan: Scan topic message using Lidar sensor, which the topic name is '/scan'
        
        Return:
        - state: arrays of processed scan_range data and heading, current_distance which means the distance between robot and goal
                
        - done: whether minimum value of scan data is negative, which means the robot collides obstacle.
    
        '''
        scan_range = []
        heading = self.heading
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        # Its
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        
        # If current_distance is lower than 0.2, we can say the robot reaches the goal.
        if current_distance < 0.2:
            self.get_goalbox = True

        # Return arrays, done
        return scan_range + [heading, current_distance], done

    def setReward(self, state, done, action):
        '''
        Design the reward function according to conditions.
        
        Parameters:
        - state: It is achieved through getState(scan) function, which consists of [scan_range, heading, current_distance] array
        - done: whether robot crashes obstacle
        - action: It is achieved through getAction(state) function in dqn_stage file, which is decided by Q-function
        
        Return:
        - reward: It is calculated by following

        '''        
        yaw_reward = []
        current_distance = state[-1]    # current_distance component of the state array from getState(scan)
        heading = state[-2]             # heading componet of the state array from getState(scan) 

        # Reward function is design follwing https://www.youtube.com/watch?time_continue=118&v=807_cByUBSI
        for step in range(5):
            angle = -math.pi / 4 + heading + (math.pi / 8 * step) + math.pi / 2
            
            # math.fabs(x): Return the absolute value of x.
            # math.modf(x): Return the fractional and integer parts of x. Both results carry the sign of x and are floats. modf[0]: decimal value, modf[1]: integer value
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            
            # yaw_reward is respectively calculated follwing five actions
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if done:
            rospy.loginfo("Collision!!")
            reward = -200
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            
            # getPosition(): reassign goal position and respawn model in Gazebo
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action):
        '''
        Publish "cmd_vel" topic to move the robot and get states.
        
        Parameters:
        - action: It is achieved through getAction(state) function in dqn_stage file, which is decided by Q-function
        
        Return:
        - np.asarray(state): state array is composed of scan_range data, heading, current_distance.  
        - reward: reward acieved from setReward(state, done, action) function 
        - done: whether robot crashes obstacle

        '''   

        max_angular_vel = 1.5

        # action_size: 5, decide to angular velocity according to action(0~4) which has maximum Q-function
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        # numpy.asarray(a, dtype=None, order=None): Convert the input to an array.
        return np.asarray(state), reward, done


    def reset(self):
        '''
        Reset GAZEBO environment
        
        Parameters:
        - 
        
        Return:
        - np.asarray(state): state array is composed of scan_range data, heading, current_distance.  

        '''   
        
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)