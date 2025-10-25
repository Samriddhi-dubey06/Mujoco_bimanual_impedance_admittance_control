import numpy as np
import mujoco
import mujoco.viewer
import time
from scipy.optimize import linprog

# Load dual-arm model
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

def get_dof_indices(model, joint_names):
    idx = []
    for name in joint_names:
        j_id = model.joint(name).id
        adr = model.jnt_dofadr[j_id]
        cnt = 3 if model.jnt_type[j_id] == mujoco.mjtJoint.mjJNT_BALL else 1
        idx.extend(range(adr, adr + cnt))
    return idx

left_arm_dofs = get_dof_indices(model, [f"1_joint_{i+1}" for i in range(6)])
right_arm_dofs = get_dof_indices(model, [f"2_joint_{i+1}" for i in range(6)])

q_des_1 = np.array([0.01099, 0.80021, -0.25865, -1.53952, 3.12771, 2.12357])
# q_des_1 = np.array([0.03560603 , 0.80020726 ,-0.25960237, -1.52043858 , 3.11171011 , 2.12256765])
q_des_2 = np.array([-0.01274, 0.80620, -0.25188, -1.52644, -0.00143, 1.02087])
# q_des_2 = np.array([-0.03555714 , 0.8079352 , -0.24859211 ,-1.52364399,  0.0028427 ,  1.02266723])

data.qpos[left_arm_dofs] = q_des_1
data.qpos[right_arm_dofs] = q_des_2
data.qvel[:] = 0
mujoco.mj_forward(model, data)

# Parameters
mu = 1.5
eta_min = 0.001
box_mass = 2
gravity = 9.81


def compute_grasp_matrix(contact_points, box_com):
    G = []
    for p in contact_points:
        r = p - box_com
        skew = np.array([[0, -r[2], r[1]],
                         [r[2], 0, -r[0]],
                         [-r[1], r[0], 0]])
        Gi = np.vstack((np.eye(3), skew))
        G.append(Gi)
    return np.hstack(G)

def build_friction_constraints(normals, mu, f_dim=3):
    A = []
    for n in normals:
        t1 = np.array([n[1], -n[0], 0])
        if np.linalg.norm(t1) < 1e-5:
            t1 = np.array([1, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        A_i = np.array([
            t1 - mu * n,
            -t1 - mu * n,
            t2 - mu * n,
            -t2 - mu * n,
            -n
        ])
        A.append(A_i)
    A_block = np.zeros((len(A)*5, f_dim * len(A)))
    for i, Ai in enumerate(A):
        A_block[i*5:(i+1)*5, i*f_dim:(i+1)*f_dim] = Ai
    return A_block

print("Robots initialized. Solving LP for optimal contact forces.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step1(model, data)

        # Freeze joint positions
        data.qpos[left_arm_dofs] = q_des_1
        data.qpos[right_arm_dofs] = q_des_2
        data.qvel[left_arm_dofs] = 0
        data.qvel[right_arm_dofs] = 0
        mujoco.mj_forward(model, data)

        tau = np.zeros(model.nu)

        contact_names = ["1_cylinder_geom", "2_cylinder_geom"]
        contact_positions = []
        for name in contact_names:
            geom_id = model.geom(name).id
            pos = data.geom_xpos[geom_id].copy()
            contact_positions.append(pos)

        box_center = data.site_xpos[model.site("box_center").id]

        # --- Read sensor forces ---
        left_id = model.sensor("left_grasp_force").id
        right_id = model.sensor("right_grasp_force").id
        left_adr = model.sensor_adr[left_id]
        right_adr = model.sensor_adr[right_id]

        f_left = data.sensordata[left_adr:left_adr + 3].copy()
        f_right = data.sensordata[right_adr:right_adr + 3].copy()

        # --- Compute normal directions from measured forces ---
        contact_normals = []
        sensor_forces = [f_left, f_right]
        min_force_threshold = 0.05

        for i, f in enumerate(sensor_forces):
            norm_f = np.linalg.norm(f)
            if norm_f > min_force_threshold:
                n = f / norm_f
            else:
                pos = contact_positions[i]
                direction = box_center - pos
                direction /= np.linalg.norm(direction)
                n = direction
            contact_normals.append(n)

        G = compute_grasp_matrix(np.array(contact_positions), box_center)
        gravity_wrench = np.array([0, 0, -box_mass * gravity, 0, 0, 0]).reshape(-1, 1)
        G_aug = np.hstack([G, gravity_wrench])

        print("[DEBUG] Full G (augmented):\n", G_aug)
        print("[DEBUG] G shape:", G_aug.shape)
        print("[DEBUG] Rank(G):", np.linalg.matrix_rank(G_aug))

        A_ineq = build_friction_constraints(contact_normals, mu)
        #  EXPLICITLY add w_ext to the RHS — to create Gλ + 2w_ext = 0
        b_eq = - gravity_wrench.flatten()

        c = np.zeros(G_aug.shape[1] + 1)
        c[-1] = -1

        A_ineq_padded = np.hstack((A_ineq, np.zeros((A_ineq.shape[0], 1))))
        A_lp = np.hstack((A_ineq_padded, -np.ones((A_ineq.shape[0], 1))))
        A_eq = np.hstack((G_aug, np.zeros((6, 1))))

        max_force = 5.0  # or even 5.0
        bounds = [(-max_force, max_force)] * G_aug.shape[1] + [(0, None)]

        res = linprog(c, A_ub=-A_lp, b_ub=-eta_min*np.ones(A_ineq.shape[0]),
                      A_eq=A_eq, b_eq=np.zeros(6), bounds=bounds, method='highs')

        if res.success:
            lambda_star = res.x[:G.shape[1]]
            f1_opt, f2_opt = lambda_star[:3], lambda_star[3:6]
            print(f"[LP] Optimal η: {res.x[-1]:.4f}")
        else:
            print("[LP ERROR] Could not solve LP. Applying fallback pressing forces.")
            f1_opt = f2_opt = np.zeros(3)

        # Fallback pressing force
        baseline_press = 2.0
        for i, direction in enumerate(contact_normals):
            if np.linalg.norm([f1_opt, f2_opt][i]) < 1e-4:
                press_force = direction * baseline_press
                if i == 0:
                    f1_opt = press_force
                else:
                    f2_opt = press_force

        for i, (dof_idx, f_des) in enumerate(zip([left_arm_dofs, right_arm_dofs], [f1_opt, f2_opt])):
            f_wrench = np.hstack([f_des, np.zeros(3)])
            geom_id = model.geom(contact_names[i]).id
            J = np.zeros((6, model.nv))
            mujoco.mj_jacGeom(model, data, J[:3], J[3:], geom_id)
            J_arm = J[:, dof_idx]
            tau_arm = J_arm.T @ f_wrench
            tau[dof_idx] = tau_arm
            print(f"[Applied] Arm {i+1} force: {f_des}, norm: {np.linalg.norm(f_des):.3f}")

        data.ctrl[:] = tau
        mujoco.mj_step2(model, data)

        print(f"[Sensor] Left: {f_left}, norm: {np.linalg.norm(f_left):.4f}")
        print(f"[Sensor] Right: {f_right}, norm: {np.linalg.norm(f_right):.4f}")

        viewer.sync()
