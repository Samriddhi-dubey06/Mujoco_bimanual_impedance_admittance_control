import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():

        mujoco.mj_step(model, data)
        viewer.sync()

# import mujoco

# MODEL_PATH = "scene.xml"
# model = mujoco.MjModel.from_xml_path(MODEL_PATH)
# data  = mujoco.MjData(model)

# # find the sensor IDs
# sid_left  = model.sensor("left_grasp_force").id
# sid_right = model.sensor("right_grasp_force").id

# # query the “frame” enum for each (0=local, 1=global, 2=object, 3=parent)
# frame_names = ["local", "global", "object", "parent"]
# fidx_left  = model._model.sensor_frame[sid_left]
# fidx_right = model._model.sensor_frame[sid_right]

# print(f"left_grasp_force  → {frame_names[fidx_left]}")
# print(f"right_grasp_force → {frame_names[fidx_right]}")

# with mujoco.viewer.launch(model, data) as viewer:
#     while viewer.is_running():
#         mujoco.mj_step(model, data)
#         viewer.sync()
