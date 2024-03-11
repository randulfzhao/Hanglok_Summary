#!/usr/bin/env python3
import rospy
import moveit_commander
import sys
from std_msgs.msg import Float64MultiArray

class UR5Controller:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('ur5_controller', anonymous=True)

        # 初始化MoveIt!界面
        moveit_commander.roscpp_initialize(sys.argv)

        # 创建机械臂的MoveGroupCommander对象
        self.arm_group = moveit_commander.MoveGroupCommander("manipulator")

        # 创建ROS订阅者，用于接收目标关节角度信息
        self.joint_angles_sub = rospy.Subscriber("target_joint_angles", Float64MultiArray, self.target_callback)

    def target_callback(self, data):
        # 回调函数，接收并处理目标关节角度信息
        target_joint_angles = data.data
        self.move_to_target(target_joint_angles)

    def move_to_target(self, target_joint_angles):
        # 设置机械臂目标关节角度
        self.arm_group.set_joint_value_target(target_joint_angles)

        # 规划和执行运动
        self.arm_group.go(wait=True)

    def shutdown(self):
        # 关闭MoveIt!界面
        moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    try:
        # 创建UR5Controller对象
        ur5_controller = UR5Controller()

        # 进入ROS循环
        rospy.spin()

        # 关闭MoveIt!界面
        ur5_controller.shutdown()

    except rospy.ROSInterruptException:
        pass
