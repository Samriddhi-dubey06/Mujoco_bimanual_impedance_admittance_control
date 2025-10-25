import numpy as np
import mujoco
import mujoco.viewer
import warnings
from qpsolvers.warnings import SparseConversionWarning

from mink import Configuration, solve_ik, SE3
from mink.tasks import FrameTask
from mink.lie import SO3

warnings.filterwarnings("ignore", category=SparseConversionWarning)

# Load dual-arm model
model = mujoco.MjModel.from_xml_path("/home/samriddhi/VelGrasp/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Utility to get DOF indices
def get_dof_indices(model, joint_names):
    idx = []
    for name in joint_names:
        j_id = model.joint(name).id
        adr = model.jnt_dofadr[j_id]
        cnt = 3 if model.jnt_type[j_id] == mujoco.mjtJoint.mjJNT_BALL else 1
        idx.extend(range(adr, adr + cnt))
    return idx

# Get full SE3 pose (position + orientation) from a site
def get_site_pose(model, data, site_name):
    site_id = model.site(site_name).id
    pos = data.site_xpos[site_id]
    rotmat = data.site_xmat[site_id].reshape(3, 3)
    rot = SO3.from_matrix(rotmat)
    return SE3.from_rotation_and_translation(rot, pos)

# Define each arm
arms = {
    "1_": {
        "joint_names": [f"1_joint_{i+1}" for i in range(6)],
        "ee_site": "1_contact_point",
        "pregrasp_site": "left_pregrasp_site",
        "grasp_site": "left_grasp_site"
    },
    "2_": {
        "joint_names": [f"2_joint_{i+1}" for i in range(6)],
        "ee_site": "2_contact_point",
        "pregrasp_site": "right_pregrasp_site",
        "grasp_site": "right_grasp_site"
    }
}

# IK parameters
dt = 0.002
damping = 1e-1
max_dq = 1.0
default_pos_threshold = 1e-3
default_ori_threshold = 1e-3
smoothing_alpha = 0.05  # Lower = smoother motion

# Initialize
q_des = data.qpos.copy()
for arm in arms.values():
    arm["dof_indices"] = get_dof_indices(model, arm["joint_names"])
    arm["stage"] = "pregrasp"
    arm["has_reached_pregrasp"] = False
    arm["has_reached_grasp"] = False

print("Running dual-arm IK with smooth control...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step1(model, data)
        data.qpos[:] = q_des
        mujoco.mj_forward(model, data)

        for arm_name, arm in arms.items():
            dof_idx = arm["dof_indices"]

            # Set thresholds
            if arm_name == "1_":
                pos_threshold = 5e-2
                ori_threshold = 5e-2
            else:
                pos_threshold = default_pos_threshold
                ori_threshold = default_ori_threshold

            # Target site
            target_site = arm["pregrasp_site"] if arm["stage"] == "pregrasp" else arm["grasp_site"]
            target_pose = get_site_pose(model, data, target_site)

            config = Configuration(model)
            config.update(q=q_des.copy())

            task = FrameTask(
                frame_name=arm["ee_site"],
                frame_type="site",
                position_cost=10.0,
                orientation_cost=1.0,
                lm_damping=1.0
            )
            task.set_target(target_pose)

            reached = False
            for _ in range(50):
                dq = solve_ik(config, tasks=[task], dt=dt, solver="osqp", damping=damping)
                dq = np.clip(dq, -max_dq, max_dq)
                q_des[dof_idx] += dq[dof_idx] * dt
                config.update(q=q_des.copy())

                err = task.compute_error(config)
                pos_err = np.linalg.norm(err[:3])
                ori_err = np.linalg.norm(err[3:])

                if arm_name == "1_":
                    print(f"[Left Arm] Stage: {arm['stage']}, pos_err: {pos_err:.4f}, ori_err: {ori_err:.4f}")

                if (arm["stage"] == "pregrasp" and pos_err < pos_threshold) or \
                   (arm["stage"] == "grasp" and pos_err < pos_threshold and ori_err < ori_threshold):
                    reached = True
                    break

            # Stage transition
            if arm["stage"] == "pregrasp" and reached and not arm["has_reached_pregrasp"]:
                print(f"{arm_name} reached pregrasp site.")
                arm["has_reached_pregrasp"] = True
                arm["stage"] = "grasp"
            elif arm["stage"] == "grasp" and reached and not arm["has_reached_grasp"]:
                print(f"{arm_name} reached grasp site.")
                arm["has_reached_grasp"] = True

            # Smooth movement command
            current_q = data.qpos[dof_idx]
            target_q = q_des[dof_idx]
            q_smooth = (1 - smoothing_alpha) * current_q + smoothing_alpha * target_q
            data.ctrl[dof_idx] = q_smooth

        mujoco.mj_step2(model, data)
        viewer.sync()

print("Both arms completed smooth pregrasp and grasp.")
