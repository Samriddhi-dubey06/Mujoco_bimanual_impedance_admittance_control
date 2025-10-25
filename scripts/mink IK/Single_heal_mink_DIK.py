import numpy as np
import mujoco
import mujoco.viewer
import warnings
from qpsolvers.warnings import SparseConversionWarning

from mink import Configuration, solve_ik, SE3
from mink.tasks import FrameTask
from mink.lie import SO3
from scipy.spatial.transform import Rotation as R  # for quat conversion

warnings.filterwarnings("ignore", category=SparseConversionWarning)

# Load model and data
model = mujoco.MjModel.from_xml_path("robot_description/heal.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Joint and actuator names
joint_names = [f"joint_{i+1}" for i in range(6)]
actuator_names = ["turret", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"]

def get_dof_indices(model, names):
    idx = []
    for n in names:
        j_id = model.joint(n).id
        adr = model.jnt_dofadr[j_id]
        cnt = 3 if model.jnt_type[j_id] == mujoco.mjtJoint.mjJNT_BALL else 1
        idx.extend(range(adr, adr + cnt))
    return idx

dof_indices = get_dof_indices(model, joint_names)
actuator_ids = [model.actuator(n).id for n in actuator_names]

# Control parameters
dt = 0.002
damping = 5e-2
max_dq = 0.5  # clamp joint velocities

# Target pose (position + orientation)
target_pos = np.array([0.3775, -0.0003, 0.5784])
quat_xyzw = np.array([0.8627, 0.5057, 0.0, 0.0004])  # [x, y, z, w] format from scipy

# Convert to rotation matrix via scipy, then to SO3
rotmat = R.from_quat(quat_xyzw).as_matrix()
target_rot = SO3.from_matrix(rotmat)
target_tf = SE3.from_rotation_and_translation(target_rot, target_pos)

# Initialize desired joint position
q_des = data.qpos.copy()

print("🔧 Running IK + position+orientation control to home target...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step1(model, data)

        # Get EE pose error
        ee_pos = data.site("contact_point").xpos.copy()
        err = np.linalg.norm(ee_pos - target_pos)

        if err < 0.005:
            data.ctrl[:] = q_des[dof_indices]
            mujoco.mj_step2(model, data)
            viewer.sync()
            continue

        # Setup configuration and IK task
        config = Configuration(model)
        config.update(q=data.qpos.copy())

        task = FrameTask(
            frame_name="contact_point",
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,  # ✅ Enable orientation tracking
            lm_damping=1.0
        )
        task.set_target(target_tf)

        # Solve IK
        dq = solve_ik(config, tasks=[task], dt=dt, solver="osqp", damping=damping)
        dq = np.clip(dq, -max_dq, max_dq)

        # Integrate and apply
        q_des[dof_indices] += dq[dof_indices] * dt
        data.ctrl[:] = q_des[dof_indices]

        mujoco.mj_step2(model, data)
        viewer.sync()

print("🏁 Target pose (position + orientation) reached and held.")
