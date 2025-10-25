import numpy as np

class ObjectAdmittanceController:
    def __init__(self, model):
        self.model = model

    def compute_object_admittance_control(self, data, left_wrench, right_wrench, desired_height=0.5):
        """
        Compute object admittance control using the equation:
        ẋₒ* = ẋₒ + D⁻¹[W - W* - K(xₒ* - xₒ)]
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            left_wrench: Wrench (force/torque) from left end-effector (6×1 vector)
            right_wrench: Wrench (force/torque) from right end-effector (6×1 vector)
            desired_height: Desired height for the object (in meters)
            
        Returns:
            object_velocity: Computed desired object velocity (6×1 vector)
        """
        # Find box body ID
        box_body_id = -1
        for i in range(self.model.nbody):
            if "box" in self.model.body(i).name:
                box_body_id = i
                break
        
        if box_body_id == -1:
            print("⚠️ Could not find box body")
            return np.zeros(6)
        
        # 1. Get current object pose
        current_obj_pos = data.xpos[box_body_id].copy()
        current_obj_quat = data.xquat[box_body_id].copy()
        
        # 2. Define desired object pose (same position but higher z)
        desired_obj_pos = current_obj_pos.copy()
        desired_obj_pos[2] = desired_height  # Set desired height
        desired_obj_quat = current_obj_quat.copy()  # Keep same orientation
        
        # 3. Compute position error
        pos_error = desired_obj_pos - current_obj_pos
        
        # Calculate distance to target height
        height_error = desired_height - current_obj_pos[2]
        height_error_percentage = min(100, max(0, (height_error / (desired_height - 0.1)) * 100))
        
        # Check if we're very close to the target height
        height_threshold = 0.01  # 1cm threshold
        if height_error < height_threshold:
            print(f"⚠️ Very close to target height! Error: {height_error:.4f}m")
            # Return minimal velocity to gently approach target
            # return np.array([0, 0, 0.02, 0, 0, 0])
        
        # 4. Get current object velocity (spatial velocity in world frame)
        current_obj_vel = np.zeros(6)
        if hasattr(data, 'cvel'):
            # Linear velocity is in the first 3 elements, angular in the last 3
            current_obj_vel[:3] = data.cvel[box_body_id, :3]
            current_obj_vel[3:] = data.cvel[box_body_id, 3:6]
        else:
            # If cvel not available, estimate from qvel
            # This is a simplified approximation
            current_obj_vel = np.zeros(6)
        
        # 5. Combine wrenches from both end-effectors
        W = left_wrench + right_wrench
        
        # 6. Define desired wrench (W*)
        # For lifting, we want to counteract gravity plus add upward force
        object_mass = self.model.body_mass[box_body_id]
        W_desired = np.zeros(6)
        
        # Calculate weight force
        weight_force = object_mass * 9.81  # N
        
        # Scale the upward force based on remaining distance to target
        # More force when far from target, less as we approach
        # Significantly increase the upward force factor to overcome weight
        upward_force_factor = 20.0 + 15.0 * (height_error_percentage / 100)
        
        # Set desired wrench to counteract gravity plus additional upward force
        # Multiply weight by a safety factor to ensure sufficient lifting force
        W_desired[2] = weight_force * 2.0 + upward_force_factor  # 2x gravity + additional force
        
        # Print weight and desired lifting force for debugging
        print(f"Object mass: {object_mass:.3f}kg, Weight: {weight_force:.3f}N")
        print(f"Desired lifting force: {W_desired[2]:.3f}N (safety factor: {W_desired[2]/weight_force:.1f}x weight)")
        
        # 7. Define stiffness (K) and damping (D) matrices
        # These are diagonal matrices for simplicity
        # Increase stiffness for more aggressive position control
        K = np.diag([200.0, 200.0, 400.0, 20.0, 20.0, 20.0])  # Stiffness (increased for z-axis)
        D = np.diag([20.0, 20.0, 10.0, 3.0, 3.0, 3.0])        # Damping (decreased for faster response)
        
        # 8. Compute pose error vector (6×1)
        pose_error = np.zeros(6)
        pose_error[:3] = pos_error
        
        
        # 9. Compute the admittance control law
        # ẋₒ* = ẋₒ + D⁻¹[W - W* - K(xₒ* - xₒ)]
        D_inv = np.linalg.inv(D)
        stiffness_term = K @ pose_error
        wrench_term = W - W_desired
        
        # Print actual vs desired wrench
        print(f"Actual wrench (z): {W[2]:.3f}N, Desired wrench (z): {W_desired[2]:.3f}N")
        print(f"Wrench difference (z): {wrench_term[2]:.3f}N")
        
        # Complete admittance control law
        object_velocity = current_obj_vel + D_inv @ (wrench_term - stiffness_term)
        
        # 10. Apply velocity limits for safety
        max_lin_vel = 0.5  # m/s (increased for faster movement)
        max_ang_vel = 0.5  # rad/s
        object_velocity[:3] = np.clip(object_velocity[:3], -max_lin_vel, max_lin_vel)
        object_velocity[3:] = np.clip(object_velocity[3:], -max_ang_vel, max_ang_vel)
        
        # For lifting, ensure we have a minimum upward velocity if not at desired height
        if height_error > 0.01:  # If we need to move up
            # Scale minimum velocity based on distance to target
            min_upward_vel = 0.2 * (height_error_percentage / 100 + 0.5)  # At least 10-20 cm/s upward
            object_velocity[2] = max(object_velocity[2], min_upward_vel)
        else:
            # Near target, slow down to avoid overshooting
            object_velocity[2] = min(object_velocity[2], 0.05)
        
        # Print debug information
        print(f"Height error: {height_error:.4f} m ({height_error_percentage:.1f}%)")
        print(f"Desired object velocity: {object_velocity}")
        
        return object_velocity