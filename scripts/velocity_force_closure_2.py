import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.optimize import linprog
from scipy.linalg import null_space
from utils.mujoco_velocity_controller.samriddhi_dual_arm_velocity_controller import VelocityControllerGC

# ——————————————— Setup ———————————————
MODEL_PATH = "scene.xml"
model      = mujoco.MjModel.from_xml_path(MODEL_PATH)
data       = mujoco.MjData(model)
controller = VelocityControllerGC(model, data)
mujoco.set_mjcb_control(controller.control_callback)

# disable gravity until after grasp
model.opt.gravity[:] = [0, 0, 0]

# desired home poses
q_des_1 = np.array([0.01099, 0.80021, -0.25865, -1.53952, 3.12771, 2.12357])
q_des_2 = np.array([-0.01274, 0.80620, -0.25188, -1.52644, -0.00143, 1.02087])

# map joint names to qpos indices and set home
left_arm_dofs  = [model.joint(f"1_joint_{i+1}").id for i in range(6)]
right_arm_dofs = [model.joint(f"2_joint_{i+1}").id for i in range(6)]
for dofs, q_des in [(left_arm_dofs, q_des_1), (right_arm_dofs, q_des_2)]:
    for j, q in zip(dofs, q_des):
        data.qpos[model.jnt_qposadr[j]] = q
mujoco.mj_forward(model, data)

# friction & LP params
mu       = 1.5
eta_min  = 0.01
box_mass = 0.5
K_diag   = 500.0

# helper: build G matrix
def compute_rigid_G(contact_positions, box_com):
    Gs = []
    for p in contact_positions:
        r = p - box_com
        skew = np.array([
            [0,    -r[2], r[1]],
            [r[2],    0, -r[0]],
            [-r[1], r[0],    0]
        ])
        Gi = np.block([
            [np.eye(3), np.zeros((3,3))],
            [skew,      np.eye(3)]
        ])  # 6×6
        Gs.append(Gi)
    return np.hstack(Gs)  # 6×12

# helper: friction pyramid + positive‐normal
def build_friction_constraints(contact_positions, normals, mu):
    A_blocks = []
    for p, n in zip(contact_positions, normals):
        # build two tangents orthonormal to n
        if abs(n[0]) < 0.9:
            v = np.array([1.0, 0.0, 0.0])
        else:
            v = np.array([0.0, 1.0, 0.0])
        t1 = v - np.dot(v, n)*n
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        A_i = np.vstack([
            ( mu*n -  t1),   #  t1·f ≤ μ n·f - η
            ( mu*n +  t1),   # -t1·f ≤ μ n·f - η
            ( mu*n -  t2),   #  t2·f ≤ μ n·f - η
            ( mu*n +  t2),   # -t2·f ≤ μ n·f - η
            (      +  n )    #  n·f ≥ η
        ])
        A_blocks.append(A_i)

    A = np.zeros((5*len(A_blocks), 6*len(A_blocks)))
    for i, Ai in enumerate(A_blocks):
        A[5*i:5*i+5, 6*i:6*i+3] = Ai
    return A

# per-contact wrench bounds
F_max = 10.0
T_max =  5.0
per_contact_bounds = [
    (-F_max, F_max),  # fx
    (-F_max, F_max),  # fy
    (    0.0, F_max), # fz ≥ 0
    (-T_max, T_max),  # τx
    (-T_max, T_max),  # τy
    (-T_max, T_max),  # τz
]
bounds = per_contact_bounds * 2 + [(0.0, None)]  # + η ≥ 0

# main loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step1(model, data)

        # contact sites & box COM
        contact_geoms    = ["1_cylinder_geom", "2_cylinder_geom"]
        contact_positions = [data.geom_xpos[model.geom(name).id].copy()
                             for name in contact_geoms]
        box_com = data.site_xpos[model.site("box_center").id]

        # 1) read local‐frame sensor forces
        left_id  = model.sensor("left_grasp_force").id
        right_id = model.sensor("right_grasp_force").id
        floc = data.sensordata[model.sensor_adr[left_id]: model.sensor_adr[left_id]+3].copy()
        rloc = data.sensordata[model.sensor_adr[right_id]:model.sensor_adr[right_id]+3].copy()
        print(f"[RAW-LOCAL] f_left={floc}, f_right={rloc}")

        # 2) rotate into world‐frame
        lsid = model.site("1_contact_point").id
        rsid = model.site("2_contact_point").id
        lbody = model.site_bodyid[lsid]
        rbody = model.site_bodyid[rsid]
        Rl = data.xmat[lbody].reshape(3,3)
        Rr = data.xmat[rbody].reshape(3,3)
        f_left  = Rl @ floc
        f_right = Rr @ rloc
        print(f"[RAW-WORLD] f_left={f_left}, f_right={f_right}")

        # 3) compute normals from sensors (fallback to geometry if zero)
        normals = []
        for i,f in enumerate((f_left, f_right)):
            norm = np.linalg.norm(f)
            if norm > 1e-6:
                n = f / norm
                print(f"[NORMAL] sensor-based normal {i+1}: {n}")
            else:
                fb = box_com - contact_positions[i]
                n  = fb / np.linalg.norm(fb)
                print(f"[NORMAL] fallback geometry normal {i+1}: {n}")
            normals.append(n)

        # 4) log true compressive terms
        for i,(f,n) in enumerate(zip((f_left,f_right), normals), start=1):
            fn = np.dot(n, f)
            print(f"[COMPRESSIVE] contact {i}: n·f = {fn:.3f}")

        # 5) build wrench‐balance
        G     = compute_rigid_G(contact_positions, box_com)
        w_ext = np.array([0,0,-9.81*box_mass, 0,0,0])

        # 6) build friction constraints
        A_ineq     = build_friction_constraints(contact_positions, normals, mu)
        A_ineq_aug = np.hstack([A_ineq, -np.ones((A_ineq.shape[0],1))])

        # 7) solve LP
        c    = np.zeros(13); c[-1] = -1.0
        A_eq = np.hstack([G, np.zeros((6,1))])
        b_eq = -w_ext

        res = linprog(
            c,
            A_ub=-A_ineq_aug, b_ub=-eta_min*np.ones(A_ineq.shape[0]),
            A_eq=A_eq,      b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        if not res.success:
            print("[LP] strict η infeasible, retry with η=0")
            res = linprog(
                c,
                A_ub=-A_ineq_aug, b_ub=np.zeros(A_ineq.shape[0]),
                A_eq=A_eq,      b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )

        if res.success:
            λ       = res.x[:12].reshape(2,6)
            η_value = res.x[-1]
            print(f"[LP] η={η_value:.4f}, λ=\n{λ}")
        else:
            print("[LP] both attempts failed – falling back to measured")
            λ = np.vstack([f_left, f_right])

        # 8) compute velocity command
        λ_vec = λ.ravel()
        # pad measured wrench to 12 elements
        measured_wrench = np.concatenate((f_left, f_right, np.zeros(6)))
        K     = np.eye(12)*K_diag
        v_c   = -np.linalg.inv(K) @ (λ_vec - measured_wrench)

        JHs = []
        for name in contact_geoms:
            J = np.zeros((6, model.nv))
            mujoco.mj_jacGeom(model, data, J[:3], J[3:], model.geom(name).id)
            JHs.append(J)
        J_H = np.vstack(JHs)

        A   = np.linalg.pinv(G.T) @ J_H
        Z   = null_space(A)
        M   = J_H @ Z

        if v_c.shape[0] < M.shape[0]:
            v_c = np.pad(v_c, (0, M.shape[0]-v_c.shape[0]))

        w_star = np.linalg.pinv(M) @ v_c
        q_dot  = Z @ w_star

        controller.set_velocity_target(q_dot[:model.nu])
        print(f"[CMD] q_dot = {q_dot[:model.nu]}")

        mujoco.mj_step2(model, data)
        viewer.sync()
