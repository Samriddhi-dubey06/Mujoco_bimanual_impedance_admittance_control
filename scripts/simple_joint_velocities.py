#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Parameters
MAX_JOINT_VEL = 1.0  # rad/s
DURATION = 60.0       # seconds to reach target

# Arbitrary initial and target joint positions (6 DOF)
q_init_A = np.array([0.0, 0.5, -0.3, 1.0, -0.2, 0.1])
q_target_A = np.array([0.2, 0.7, -0.1, 1.2, -0.1, 0.3])

q_init_B = np.array([-0.1, 0.4, -0.2, 0.9, -0.3, 0.0])
q_target_B = np.array([0.1, 0.6, 0.0, 1.1, -0.1, 0.2])

# Velocity placeholders
q_dot_A = (q_target_A - q_init_A) / DURATION
q_dot_B = (q_target_B - q_init_B) / DURATION

# Clamp to robot limits
q_dot_A = np.clip(q_dot_A, -MAX_JOINT_VEL, MAX_JOINT_VEL)
q_dot_B = np.clip(q_dot_B, -MAX_JOINT_VEL, MAX_JOINT_VEL)

# ---- ROS Publishers ----
def publish_velocities():
    rospy.init_node("dual_arm_constant_velocity")

    pubA = rospy.Publisher("/robotA/velocity_controller/command", Float64MultiArray, queue_size=1)
    pubB = rospy.Publisher("/robotB/velocity_controller/command", Float64MultiArray, queue_size=1)

    rate = rospy.Rate(100)
    start_time = rospy.get_time()

    while not rospy.is_shutdown():
        elapsed = rospy.get_time() - start_time
        if elapsed >= DURATION:
            rospy.loginfo("Target reached, stopping...")
            pubA.publish(Float64MultiArray(data=[0.0]*6))
            pubB.publish(Float64MultiArray(data=[0.0]*6))
            break

        pubA.publish(Float64MultiArray(data=q_dot_A.tolist()))
        pubB.publish(Float64MultiArray(data=q_dot_B.tolist()))

        rospy.loginfo_throttle(1.0, f"Sending velocities A: {q_dot_A.round(3)} | B: {q_dot_B.round(3)}")
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_velocities()
    except rospy.ROSInterruptException:
        pass
