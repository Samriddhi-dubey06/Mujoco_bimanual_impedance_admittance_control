import numpy as np
import mujoco
import mujoco.viewer
import time
from scipy.optimize import linprog
from scipy.linalg import null_space

# Load model
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
from utils.mujoco_velocity_controller.samriddhi_dual_arm_velocity_controller import VelocityControllerGC
controller = VelocityControllerGC(model, data)
mujoco.set_mjcb_control(controller.control_callback)

# Disable gravity initially
model.opt.gravity[:] = [0, 0, 0]

# Joint setup
left_arm_dofs = [model.joint(f"1_joint_{i+1}").id for i in range(6)]
right_arm_dofs = [model.joint(f"2_joint_{i+1}").id for i in range(6)]

# q_des_1 = np.array([0.01099, 0.80021, -0.25865, -1.53952, 3.12771, 2.12357])
# q_des_2 = np.array([-0.01274, 0.80620, -0.25188, -1.52644, -0.00143, 1.02087])
q_des_1 = np.array([0.01099, 0.80021, -0.25865, -1.53952, 3.12771, 2.12357])
q_des_2 = np.array([-0.01274, 0.80620, -0.25188, -1.52644, -0.00143, 1.02087])

# Initial pose
for i, q in zip(left_arm_dofs, q_des_1):
    data.qpos[model.jnt_qposadr[i]] = q
for i, q in zip(right_arm_dofs, q_des_2):
    data.qpos[model.jnt_qposadr[i]] = q
mujoco.mj_forward(model, data)

# Parameters
mu = 1.5
eta_min = 0.01
box_mass = 0.5
K_diag = 500.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    gravity_enabled = False
    while viewer.is_running():
        mujoco.mj_step1(model, data)

        contact_geoms = ["1_cylinder_geom", "2_cylinder_geom"]
        contact_positions = [data.geom_xpos[model.geom(name).id].copy() for name in contact_geoms]
        box_com = data.site_xpos[model.site("box_center").id]

        # Sensor readings
        left_id = model.sensor("left_grasp_force").id
        right_id = model.sensor("right_grasp_force").id
        left_adr = model.sensor_adr[left_id]
        right_adr = model.sensor_adr[right_id]

        f_left = data.sensordata[left_adr:left_adr + 3].copy()
        f_right = data.sensordata[right_adr:right_adr + 3].copy()
        print(f"[SENSOR] f_left:  {f_left},  norm: {np.linalg.norm(f_left):.4f}")
        print(f"[SENSOR] f_right: {f_right}, norm: {np.linalg.norm(f_right):.4f}")

        measured_f = np.concatenate((f_left, f_right))

        # Use geometry-based normals
        normals = []
        for i, f in enumerate([f_left, f_right]):
            norm = np.linalg.norm(f)
            if norm > 1e-6:
                n = f / norm
                print(f"[NORMAL] Sensor-based normal used for contact {i+1}: {n}")
            else:
                # fallback if force is zero — use geometry
                fallback = box_com - contact_positions[i]
                fallback /= np.linalg.norm(fallback)
                n = fallback
                print(f"[NORMAL] Sensor force too small at contact {i+1}, using fallback normal: {n}")
            normals.append(n)

        def compute_rigid_G():
            G = []
            for p in contact_positions:
                r = p - box_com
                skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
                Gi = np.block([
                    [np.eye(3), np.zeros((3,3))],
                    [skew,      np.eye(3)]
                ])  # (6,6) per contact
                G.append(Gi)
            return np.hstack(G)  # (6,12)

        G = compute_rigid_G()
        w_ext = np.array([0, 0, (-9.81 * box_mass)+2, 0, 0, 0])
        # w_ext = np.zeros(6)  # No external wrench


        def build_friction_constraints():
            A = []
            for i, n in enumerate(normals):
                f = [f_left, f_right][i]
                fx, fy, fz = f

                # Sensor-based tangent directions
                t1 = np.array([1.0, 0.0, 0.0])  # direction for fx
                t2 = np.array([0.0, 1.0, 0.0])  # direction for fy

                A_i = np.array([
                    t1 - mu * n,
                    -t1 - mu * n,
                    t2 - mu * n,
                    -t2 - mu * n,
                    -n
                ])
                A.append(A_i)

            A_block = np.zeros((len(A) * 5, 6 * len(A)))
            for i, Ai in enumerate(A):
                A_block[i * 5:(i + 1) * 5, i * 6:i * 6 + 3] = Ai  # Apply to force part
            return A_block


        A_ineq = build_friction_constraints()
        c = np.zeros(13)
        c[-1] = -1
        A_ineq_aug = np.hstack([A_ineq, -np.ones((A_ineq.shape[0], 1))])
        A_eq = np.hstack([G, np.zeros((6, 1))])
        b_eq = -w_ext
        bounds = [(-10,10)] * 12 + [(0, None)]

        print("[DEBUG] Rank of G:", np.linalg.matrix_rank(G), "/", G.shape)

        res = linprog(c, A_ub=-A_ineq_aug, b_ub=-eta_min * np.ones(A_ineq.shape[0]),
              A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        relaxed_lp = False

        if not res.success:
            print("[LP WARNING] First LP (strict η ≥ η_min) infeasible.")
            print("[LP INFO] Retrying LP with relaxed constraint (η = 0)...")

            res = linprog(c, A_ub=-A_ineq_aug, b_ub=np.zeros(A_ineq.shape[0]),
                        A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            relaxed_lp = True

        if res.success:
            lambda_star = res.x[:12]
            eta_value = res.x[-1]
            print(f"[LP] η (safety margin): {eta_value:.4f}")
            print(f"[LP] λ (lambda_star): {lambda_star.reshape(2, 6)}")
            print(f"[LP] Status: {'RELAXED LP' if relaxed_lp else 'STRICT LP'}")
        else:
            print("[LP ERROR] Both LP attempts failed — falling back to measured force.")
            lambda_star = np.zeros(12)
            lambda_star[:6] = measured_f


        K = np.eye(len(lambda_star)) * K_diag
        v_c = -np.linalg.inv(K) @ (lambda_star - np.pad(measured_f, (0, 6)))

        J_H = []
        for name in contact_geoms:
            gid = model.geom(name).id
            J = np.zeros((6, model.nv))
            mujoco.mj_jacGeom(model, data, J[:3], J[3:], gid)
            J_H.append(J)
        J_H = np.vstack(J_H)  # (12, nv)

        A = np.linalg.pinv(G.T) @ J_H
        Z = null_space(A)
        M = J_H @ Z

        if v_c.shape[0] != M.shape[0]:
            v_c = np.pad(v_c, (0, M.shape[0] - v_c.shape[0]))

        w_star = np.linalg.pinv(M) @ v_c
        q_dot = Z @ w_star

        controller.set_velocity_target(q_dot[:model.nu])
        print("[INFO] Velocity target sent to controller.")
        print(f"[DEBUG] q_dot[:model.nu]: {q_dot[:model.nu]}")

        # # Enable gravity on the box after 2 seconds
        # if not gravity_enabled and (time.time() - start_time) > 2.0:
        #     model.opt.gravity[:] = [0, 0, -9.81]
        #     gravity_enabled = True
        #     print("Gravity enabled.")


        import csv
        import os

        # Initialize log list at the start of simulation
        if 'joint_vel_log' not in locals():
            joint_vel_log = []
            log_start_time = time.time()

        # Record time and qvel
        log_time = time.time() - log_start_time
        joint_vel_log.append([log_time] + data.qvel[:model.nu].tolist())
        lambda_left_f = lambda_star[:3]
        lambda_right_f = lambda_star[6:9]

        # Combine into one row
        lambda_log_row = [log_time]
        lambda_log_row += lambda_left_f.tolist() + lambda_right_f.tolist()

        # Initialize log list if not already
        if 'lambda_log' not in locals():
            lambda_log = []

        lambda_log.append(lambda_log_row)

        # Build row for force logging
        force_log_row = [log_time]
        force_log_row += f_left.tolist() + f_right.tolist()
        force_log_row += lambda_left_f.tolist() + lambda_right_f.tolist()

        # Initialize list if needed
        if 'force_log' not in locals():
            force_log = []

        force_log.append(force_log_row)


        mujoco.mj_step2(model, data)
        # Print current joint velocities
        print(f"[STATE] qvel: {data.qvel[:model.nu]}")
        viewer.sync()

# # Save joint velocity log to CSV
# csv_path = "joint_velocity_log.csv"
# with open(csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     header = ["time"] + [f"joint_{i+1}" for i in range(model.nu)]
#     writer.writerow(header)
#     writer.writerows(joint_vel_log)

# print(f"[INFO] Joint velocities logged to {csv_path}")
# lambda_csv_path = "lambda_log.csv"
# with open(lambda_csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     header = ["time",
#               "lambda_left_fx", "lambda_left_fy", "lambda_left_fz",
#               "lambda_right_fx", "lambda_right_fy", "lambda_right_fz"]
#     writer.writerow(header)
#     writer.writerows(lambda_log)

# print(f"[INFO] Lambda values logged to {lambda_csv_path}")

force_csv_path = "force_comparison_log.csv"
with open(force_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ["time",
              "f_left_fx", "f_left_fy", "f_left_fz",
              "f_right_fx", "f_right_fy", "f_right_fz",
              "lambda_left_fx", "lambda_left_fy", "lambda_left_fz",
              "lambda_right_fx", "lambda_right_fy", "lambda_right_fz"]
    writer.writerow(header)
    writer.writerows(force_log)

print(f"[INFO] Force data logged to {force_csv_path}")

