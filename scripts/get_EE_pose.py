import mujoco
import numpy as np
from mink import SE3
from mink.lie import SO3
from scipy.spatial.transform import Rotation as R  

# Load MuJoCo model and data
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Function to get full SE3 pose of an end-effector site
def get_end_effector_pose(site_name):
    site_id = model.site(site_name).id
    pos = data.site_xpos[site_id]  # 3D position
    rotmat = data.site_xmat[site_id].reshape(3, 3)  # 3x3 rotation matrix
    rotation = SO3.from_matrix(rotmat)
    quat = R.from_matrix(rotmat).as_quat()  # [x, y, z, w]
    return pos, quat

# Get poses
pos_1, quat_1 = get_end_effector_pose("1_contact_point")
pos_2, quat_2 = get_end_effector_pose("2_contact_point")

# Print
print("🔹 End-Effector 1 (1_contact_point)")
print("Position:", pos_1)
print("Quaternion [x, y, z, w]:", quat_1)

print("\n🔹 End-Effector 2 (2_contact_point)")
print("Position:", pos_2)
print("Quaternion [x, y, z, w]:", quat_2)
