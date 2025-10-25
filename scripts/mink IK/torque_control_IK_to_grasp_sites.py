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
model = mujoco.MjModel.from_xml_path("scene.xml")
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

# IK and control parameters
dt = 0.002 # This defines how frequently control commands are updated (500 Hz)
damping = 1e-1 # Damping coefficient for inverse kinematics (IK) solver
max_dq = 1.0 # Maximum allowed joint velocity change per IK iteration (in rad/s)
default_pos_threshold = 1e-3 # Default position error threshold (in meters)
default_ori_threshold = 1e-3
K_v = 50.0  # Velocity feedback gain

# Initialize
q_des = data.qpos.copy()


for arm in arms.values():
    arm["dof_indices"] = get_dof_indices(model, arm["joint_names"])
    arm["stage"] = "pregrasp"
    arm["has_reached_pregrasp"] = False
    arm["has_reached_grasp"] = False

# # Get the indices of all robot-controlled DOFs
# robot_dof_indices = arms["1_"]["dof_indices"] + arms["2_"]["dof_indices"]
print("Running dual-arm IK with torque control...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step1(model, data)
        data.qpos[:] = q_des
        mujoco.mj_forward(model, data)

        tau = np.zeros(model.nu)

        for arm_name, arm in arms.items():
            dof_idx = arm["dof_indices"]

            # Set thresholds
            if arm_name == "1_":
                pos_threshold = 5e-3
                ori_threshold = 0.3
            else:
                pos_threshold = default_pos_threshold
                ori_threshold = default_ori_threshold

            # Target site
            target_site = arm["pregrasp_site"] if arm["stage"] == "pregrasp" else arm["grasp_site"]
            target_pose = get_site_pose(model, data, target_site)

            # IK setup
            config = Configuration(model)
            config.update(q=q_des.copy())

            task = FrameTask(
                frame_name=arm["ee_site"],
                frame_type="site",
                position_cost=10.0, #A higher value = the solver tries harder to minimize position error.
                orientation_cost=1.0, #Lower than position_cost, meaning you care more about reaching the location than matching the rotation exactly.
                lm_damping=1.0 # Levenberg-Marquardt damping (prevents large joint updates or instability, especially near singularties or wehn the goals are nearly unreachable)
            )
            task.set_target(target_pose)

            reached = False
            dq_des = np.zeros_like(data.qvel)

            for _ in range(50):
                dq = solve_ik(config, tasks=[task], dt=dt, solver="osqp", damping=damping)
                dq = np.clip(dq, -max_dq, max_dq)
                dq_des[dof_idx] = dq[dof_idx]
                q_des[dof_idx] += dq[dof_idx] * dt
                config.update(q=q_des.copy())

                err = task.compute_error(config)
                pos_err = np.linalg.norm(err[:3])
                ori_err = np.linalg.norm(err[3:])

                if arm_name == "1_":
                    print(f"[Left Arm] Stage: {arm['stage']}, pos_err: {pos_err:.4f}, ori_err: {ori_err:.4f}")

                if (arm["stage"] == "pregrasp" and pos_err < pos_threshold and ori_err < ori_threshold) or \
                   (arm["stage"] == "grasp" and pos_err < pos_threshold and ori_err < ori_threshold):
                    reached = True
                    break

            # Stage transition
            if arm["stage"] == "pregrasp" and reached and not arm["has_reached_pregrasp"]:
                arm["has_reached_pregrasp"] = True
                print(f"{arm_name} reached pregrasp site.")

            # Global synchronization: only transition when both arms are ready
            if all(a["has_reached_pregrasp"] for a in arms.values()):
                for other_arm in arms.values():
                    if other_arm["stage"] == "pregrasp":
                        other_arm["stage"] = "grasp"

            # Handle grasp transition per arm
            if arm["stage"] == "grasp" and reached and not arm["has_reached_grasp"]:
                print(f"{arm_name} reached grasp site.")
                arm["has_reached_grasp"] = True
            
                

            # Velocity error for torque control
            qvel_curr = data.qvel[dof_idx]
            dq_cmd = dq_des[dof_idx]
            vel_err = dq_cmd - qvel_curr

            # Gravity compensation torque
            mujoco.mj_rnePostConstraint(model, data)  # updates data.qfrc_bias
            gravity_torque = data.qfrc_bias[dof_idx]

            tau_force = np.zeros_like(data.qvel)


            # if arm["has_reached_grasp"]:
            #     f_des= np.array([0,0,5,0,0,0])
            #     J = np.zeros((6, model.nv))
            #     site_id=model.site(arm["ee_site"]).id
            #     mujoco.mj_jacSite(model, data, J[:3], J[3:], site_id)

            #     tau_force = J.T @ f_des
                # print(f"[{arm_name}] Applying force {f_des} at EE.")

            # Final torque command
            tau_arm = gravity_torque + K_v * vel_err 
            # print(f"[{arm_name}] Final tau_arm: {tau_arm}")

            tau[dof_idx] = tau_arm

        # Apply torques
        data.ctrl[:] = tau

        mujoco.mj_step2(model, data)
        viewer.sync()

        # Print success message during simulation
        if all(a["has_reached_grasp"] for a in arms.values()):
            print("Both arms completed smooth pregrasp and grasp using torque control.")
            # break 

