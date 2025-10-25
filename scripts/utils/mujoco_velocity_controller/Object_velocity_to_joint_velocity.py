import mujoco
import numpy as np

class VelocityMapper:
    def __init__(self, model):
        self.model = model

    def object_velocity_to_joint_velocities(self, data, object_velocity):
        """
        Convert object velocity to joint velocities for dual-arm manipulation.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            object_velocity: Desired object velocity (6×1 vector)
            
        Returns:
            joint_velocities: Joint velocities for both arms
        """
        try:
            # Find box body ID
            box_body_id = -1
            for i in range(self.model.nbody):
                if "box" in self.model.body(i).name:
                    box_body_id = i
                    break
            
            if box_body_id == -1:
                print("⚠️ Could not find box body")
                return np.zeros(12)  # Assuming 6 DOFs per arm
            
            # Find end-effector site IDs - search more broadly for site names
            left_ee_site_id = -1
            right_ee_site_id = -1
            
            # Print all available sites for debugging
            print("\n--- AVAILABLE SITES ---")
            for i in range(self.model.nsite):
                site_name = self.model.site(i).name
                print(f"Site {i}: {site_name}")
                
                # Look for end-effector sites
                if "left" in site_name.lower() and ("ee" in site_name.lower() or "end" in site_name.lower() or "gripper" in site_name.lower()):
                    left_ee_site_id = i
                elif "right" in site_name.lower() and ("ee" in site_name.lower() or "end" in site_name.lower() or "gripper" in site_name.lower()):
                    right_ee_site_id = i
            
            # If specific end-effector sites not found, use any left/right sites
            if left_ee_site_id == -1 or right_ee_site_id == -1:
                print("⚠️ Could not find specific end-effector sites, searching for any left/right sites...")
                for i in range(self.model.nsite):
                    site_name = self.model.site(i).name
                    if "left" in site_name.lower() and left_ee_site_id == -1:
                        left_ee_site_id = i
                        print(f"Using as left end-effector: {site_name}")
                    elif "right" in site_name.lower() and right_ee_site_id == -1:
                        right_ee_site_id = i
                        print(f"Using as right end-effector: {site_name}")
            
            if left_ee_site_id == -1 or right_ee_site_id == -1:
                print("⚠️ Could not find end-effector sites")
                return np.zeros(12)  # Assuming 6 DOFs per arm
            
            # Find joint IDs for each arm
            left_arm_dofs = []
            right_arm_dofs = []
            
            # Assuming the first half of actuators belong to the left arm and the second half to the right arm
            num_actuators = self.model.nu
            actuators_per_robot = num_actuators // 2
            
            # Get DOFs for left arm
            for i in range(actuators_per_robot):
                joint_id = self.model.actuator_trnid[i, 0]
                dof_adr = self.model.jnt_dofadr[joint_id]
                left_arm_dofs.append(dof_adr)
            
            # Get DOFs for right arm
            for i in range(actuators_per_robot, num_actuators):
                joint_id = self.model.actuator_trnid[i, 0]
                dof_adr = self.model.jnt_dofadr[joint_id]
                right_arm_dofs.append(dof_adr)
            
            # Get Jacobians for both end-effectors
            jac_left = np.zeros((6, self.model.nv))
            jac_right = np.zeros((6, self.model.nv))
            
            mujoco.mj_jacSite(self.model, data, jac_left[:3], jac_left[3:], left_ee_site_id)
            mujoco.mj_jacSite(self.model, data, jac_right[:3], jac_right[3:], right_ee_site_id)
            
            # Number of DOFs per arm
            n_dofs_per_arm = len(left_arm_dofs)
            total_dofs = n_dofs_per_arm * 2
            
            # Extract relevant columns from Jacobians
            jac_left_arm = jac_left[:, left_arm_dofs]
            jac_right_arm = jac_right[:, right_arm_dofs]
            
            # Combine Jacobians for both arms
            J_combined = np.zeros((12, total_dofs))
            J_combined[:6, :n_dofs_per_arm] = jac_left_arm
            J_combined[6:, n_dofs_per_arm:] = jac_right_arm
            
            # For lifting, we're mainly concerned with the z-direction
            # Extract z-rows from Jacobians
            J_z = np.zeros((2, total_dofs))
            J_z[0, :n_dofs_per_arm] = jac_left_arm[2, :]  # z-row for left arm
            J_z[1, n_dofs_per_arm:] = jac_right_arm[2, :]  # z-row for right arm
            
            # Compute pseudoinverse
            J_z_pinv = np.linalg.pinv(J_z)
            
            # Compute joint velocities for lifting
            v_z = np.array([object_velocity[2], object_velocity[2]])  # Same z-velocity for both contacts
            joint_velocities = J_z_pinv @ v_z
            
            # Scale for faster movement
            joint_velocities *= 2.0
            
            # Apply joint velocity limits
            max_joint_vel = 0.5  # rad/s
            joint_velocities = np.clip(joint_velocities, -max_joint_vel, max_joint_vel)
            
            print(f"Object velocity: {object_velocity}")
            print(f"Joint velocities: {joint_velocities}")
            
            return joint_velocities
            
        except Exception as e:
            print(f"Error in object_velocity_to_joint_velocities: {e}")
            import traceback
            traceback.print_exc()
            
            