#!/usr/bin/env python3

import sys
import moveit_commander
import rospy
import pickle
import torch

with open('/home/rzhao/code_work/catkin_ws/src/ur5_control/scripts/extracted.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    
joint_goals_list = loaded_data['A006'][3]
joint_goals_list = joint_goals_list.tolist()

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('ur5_moveit_joint_space_trajectories', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # # 定义一系列关节角度目标（以弧度为单位）
    # joint_goals_list = [
    #     [0.1, -0.5, 1.0, -1.5, -1.0, 0.5],  # 第一组关节角度
    #     [0.2, -0.6, 1.1, -1.6, -1.1, 0.6],  # 第二组关节角度
    #     # 可以根据需要添加更多的关节角度组
    # ]

    # 对于每组关节角度，规划并执行轨迹
    for joint_goals in joint_goals_list:
        move_group.go(joint_goals, wait=True)
        move_group.stop()  # 停止所有剩余的运动

        rospy.sleep(1)  # 等待一段时间，然后继续下一组关节角度

if __name__ == '__main__':
    main()


