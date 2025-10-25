#!/usr/bin/env python3
import rospy
import os
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import kdl_parser_py.urdf as kdl_urdf
from PyKDL import ChainJntToJacSolver, JntArray, Jacobian

# ---- Parameters ----
K_vec = np.array([0.0, 0.0, 400.0, 0.0, 0.0, 0.0])  # Stiffness in Z only
D_vec = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])   # Damping in Z only
MAX_JOINT_VEL = 1.0  # rad/s

# Object parameters
object_mass = 0.5  # kg
object_weight = object_mass * 9.81  # N

# URDF path & links
urdf_path = "/home/addverb/Debojit_WS/Addverb_Heal_and_Syncro_Hardware/src/cobot_ros/harmonic_urdf_7/urdf/left_arm.urdf"
base_link = "Torso"
ee_link   = "end_effector_left"

# ---- State ----
joint_positions_A = None
x_obj = np.array([0.0, 0.0, 0.2])
v_obj = np.zeros(6)
desired_obj_pos = np.array([0.0, 0.0, 0.5])

# ---- Callbacks ----
def joint_cb(msg):
    global joint_positions_A
    if len(msg.position) >= 6:
        joint_positions_A = np.array(msg.position[:6])

# ---- Helpers ----
def kdl_jacobian(chain, solver, q_vec):
    n = chain.getNrOfJoints()
    ja = JntArray(n)
    for i in range(n):
        ja[i] = q_vec[i]
    J_kdl = Jacobian(n)
    solver.JntToJac(ja, J_kdl)
    J = np.zeros((6, n))
    for r in range(6):
        for c in range(n):
            J[r, c] = J_kdl[r, c]
    return J

# ---- Main ----
def main():
    global x_obj, v_obj

    rospy.init_node('object_admittance_to_joint_velocity')

    # Load URDF and KDL chain
    if not os.path.isfile(urdf_path):
        rospy.logerr("URDF not found at: %s", urdf_path)
        return
    ok, tree = kdl_urdf.treeFromFile(urdf_path)
    if not ok:
        rospy.logerr("Failed to parse URDF")
        return
    chain = tree.getChain(base_link, ee_link)
    n = chain.getNrOfJoints()
    jac_solver = ChainJntToJacSolver(chain)

    # ROS I/O
    rospy.Subscriber('/robotA/joint_states', JointState, joint_cb)
    pubA = rospy.Publisher('/robotA/velocity_controller/command', Float64MultiArray, queue_size=1)
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        if joint_positions_A is None:
            rospy.logwarn_throttle(2.0, "Waiting for joint states")
            rate.sleep()
            continue

        # Compute object pose error
        pos_error = desired_obj_pos - x_obj
        pose_error = np.zeros(6)
        pose_error[:3] = pos_error

        # Hardcoded wrenches
        W = np.array([0.0, 0.0, 25.0, 0.0, 0.0, 0.0])
        W_des = np.array([0.0, 0.0, object_weight, 0.0, 0.0, 0.0])

        # Admittance control law: v_obj = v_obj + D^-1 * (W - W_des - K*x_error)
        D = np.diag(D_vec + 1e-6)  # regularize
        K = np.diag(K_vec)
        v_obj += np.linalg.inv(D) @ (W - W_des - K @ pose_error)
        v_obj[2] = np.clip(v_obj[2], 0.0, 0.5)  # upward only, max 0.5 m/s

        # Compute Jacobian
        Jc = kdl_jacobian(chain, jac_solver, joint_positions_A)
        if Jc.shape == (6, n):
            try:
                cond_number = np.linalg.cond(Jc)
                if cond_number > 1e4:
                    rospy.logwarn_throttle(2.0, "Jacobian near singularity (cond=%.2e), skipping", cond_number)
                    rate.sleep()
                    continue
            except np.linalg.LinAlgError:
                rospy.logerr("Jacobian error")
                rate.sleep()
                continue

            # Joint velocity from Cartesian velocity
            q_dot = np.linalg.pinv(Jc).dot(v_obj)
            q_dot = np.clip(q_dot, -MAX_JOINT_VEL, MAX_JOINT_VEL)
            pubA.publish(Float64MultiArray(data=q_dot.tolist()))

            # Euler integration of object motion
            x_obj += v_obj[:3] * 0.01
            rospy.loginfo_throttle(1.0, f"Z: {x_obj[2]:.3f} m | Vel Z: {v_obj[2]:.3f} | q_dot: {q_dot[2]:.3f}")
        else:
            rospy.logerr("Jacobian shape mismatch")

        rate.sleep()

if __name__ == "__main__":
    main()
