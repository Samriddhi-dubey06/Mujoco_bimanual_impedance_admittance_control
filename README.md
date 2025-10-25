Bimanual Impedance–Admittance Control in MuJoCo

This repository implements a bimanual robotic manipulation framework using impedance and admittance control in the MuJoCo physics simulator. The objective of this work is to achieve compliant dual-arm interaction with objects while maintaining stable force and motion control.

Overview

The project focuses on developing a hybrid control strategy that combines impedance and admittance control principles for a bimanual robot setup. It enables both arms to cooperatively manipulate an object while adapting to contact forces and environmental disturbances.

Key aspects include:
- Modeling a dual-arm robot in MuJoCo
- Implementing impedance control for motion compliance
- Implementing admittance control to translate external forces into motion adaptation
- Integrating simulated force/torque feedback for contact interaction
- Coordinating trajectories between the two manipulators for stable object manipulation

Implementation Details

- Simulation Environment: MuJoCo is used to simulate the robot dynamics and contact physics.
- Robot Model: The robot is modeled as a two-arm manipulator with joints, end-effectors, and contact sensors.
- Control Scheme:
  - Impedance control is used to regulate end-effector motion based on desired stiffness and damping parameters.
  - Admittance control adjusts the desired trajectory in response to contact forces.
  - The combined approach ensures compliant behavior even during dynamic contact conditions.
- Trajectory Generation: Cartesian trajectories for both arms are generated to perform coordinated motion while maintaining a defined relative pose between end-effectors.
- Data Logging and Visualization: The system records joint positions, end-effector forces, and tracking errors for analysis.

Results

- Demonstrated compliant dual-arm manipulation using impedance–admittance control.
- The controller maintains stability during object contact and adapts to external disturbances.
- Visualized motion and force tracking in simulation to validate controller performance.

Repository Structure

mujoco_model/
    bimanual_robot.xml       # Robot model and simulation setup
    assets/                  # MuJoCo assets (meshes, textures, etc.)
controllers/
    impedance_controller.py  # Implements impedance control logic
    admittance_controller.py # Implements admittance control logic
main_control.py              # Main simulation and control loop
config.py                    # Control parameters (stiffness, damping, etc.)
README.md

How to Run

1. Clone the repository:
   git clone https://github.com/Samriddhi-dubey06/Mujoco_bimanual_impedance_admittance_control
   cd Mujoco_bimanual_impedance_admittance_control

2. Install required dependencies:
   pip install mujoco dm_control numpy matplotlib scipy

3. Run the main control script:
   python main_control.py

4. Modify control parameters in config.py to test different compliance behaviors.

Author

Samriddhi Dubey
B.Tech, Mechanical Engineering, IIT Gandhinagar
Focus Areas: Robotics, Control Systems, Dual-Arm Manipulation
