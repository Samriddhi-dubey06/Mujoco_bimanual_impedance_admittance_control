#!/usr/bin/env python3
import rospy
import os
import csv
import numpy as np
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import kdl_parser_py.urdf as kdl_urdf
from PyKDL import ChainJntToJacSolver, JntArray, Jacobian

# ---- Parameters ----
K_VEC = np.array([1.0, 0.2, 0.1, 0.1, 0.1, 0.1])
MAX_JOINT_VEL = 0.1
FILTER_ALPHA = 0.01

# ---- State ----
f_left = np.zeros(6)
joint_positions = None

# ---- CSV setup ----
csv_folder = os.path.expanduser('~/csv')
os.makedirs(csv_folder, exist_ok=True)

joint_f = open(os.path.join(csv_folder, 'joint_velocities.csv'), 'w', newline='')
cart_f  = open(os.path.join(csv_folder, 'cartesian_velocities.csv'), 'w', newline='')
force_f = open(os.path.join(csv_folder, 'forces.csv'), 'w', newline='')

joint_w = csv.writer(joint_f)
cart_w  = csv.writer(cart_f)
force_w = csv.writer(force_f)

# write headers
joint_w.writerow(['time'] + [f'qdot_{i}' for i in range(6)])
cart_w.writerow( ['time'] + [f'v_c_{i}' for i in range(6)] )
force_w.writerow(
    ['time'] +
    [f'lambda_meas_{i}' for i in range(6)] +
    [f'lambda_star_{i}' for i in range(6)]
)

def close_files():
    joint_f.close()
    cart_f.close()
    force_f.close()
    rospy.loginfo(f'CSV files saved in {csv_folder}')

def ft_callback(msg):
    global f_left
    f_new = np.hstack(([0, 0, msg.wrench.force.z], [0,0,0]))
    f_left = FILTER_ALPHA * f_new + (1 - FILTER_ALPHA) * f_left

def joint_cb(msg):
    global joint_positions
    if len(msg.position) >= 6:
        joint_positions = np.array(msg.position[:6])

def kdl_jacobian(chain, solver, q):
    n = chain.getNrOfJoints()
    ja = JntArray(n)
    for i in range(n):
        ja[i] = q[i]
    J_kdl = Jacobian(n)
    solver.JntToJac(ja, J_kdl)
    J = np.zeros((6, n))
    for r in range(6):
        for c in range(n):
            J[r, c] = J_kdl[r, c]
    return J

def main():
    rospy.init_node('ft_logger_split')
    rospy.on_shutdown(close_files)

    # parse URDF
    urdf_path = '/home/iitgn-robotics/Samriddhi_WS/bimanual_ws/src/addverb_heal_description/urdf/robot.urdf'
    if not os.path.isfile(urdf_path):
        rospy.logerr(f'URDF not found: {urdf_path}')
        return

    ok, tree = kdl_urdf.treeFromFile(urdf_path)
    if not ok:
        rospy.logerr('Failed to parse URDF')
        return

    chain = tree.getChain('base_link', 'tool')
    jac_solver = ChainJntToJacSolver(chain)
    n = chain.getNrOfJoints()

    rospy.Subscriber('/ft_sensor', WrenchStamped, ft_callback)
    rospy.Subscriber('/joint_states', JointState, joint_cb)
    pub = rospy.Publisher('/velocity_controller/command',
                          Float64MultiArray, queue_size=1)

    rate = rospy.Rate(100)
    t0 = rospy.Time.now().to_sec()

    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t0
        lam_meas = f_left.copy()
        lam_star = np.array([0.0, 0, -5.0, 0, 0, 0])

        # decide q_dot & v_c
        if joint_positions is None:
            q_dot = np.zeros(n)
            v_c   = np.zeros(6)
        else:
            error = abs(lam_star[0] - lam_meas[0])
            if error < 0.01:
                q_dot = np.zeros(n)
                v_c   = np.zeros(6)
            else:
                v_c = (0.01*(lam_star - lam_meas) )/ K_VEC
                J  = kdl_jacobian(chain, jac_solver, joint_positions)
                if J.shape == (6,n) and np.linalg.cond(J) < 1e4:
                    q_dot = np.clip(np.linalg.pinv(J).dot(v_c),
                                    -MAX_JOINT_VEL, MAX_JOINT_VEL)
                else:
                    q_dot = np.zeros(n)

        pub.publish(Float64MultiArray(data=q_dot.tolist()))

        # log & flush each cycle
        joint_w.writerow([t] + q_dot.tolist()); joint_f.flush()
        cart_w.writerow( [t] + v_c.tolist() );      cart_f.flush()
        force_w.writerow([t] + lam_meas.tolist() + lam_star.tolist()); force_f.flush()

        rate.sleep()

if __name__ == '__main__':
    main()

