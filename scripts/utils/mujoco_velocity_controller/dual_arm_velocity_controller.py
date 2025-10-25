import numpy as np

class VelocityController:
    """
    A derivative-only (D) controller for all actuators,
    while applying gravity compensation to all joints.
    """
    def __init__(self, model, data, kd=50.0, ki=0.05):
        """r
        :param model: The MuJoCo MjModel
        :param data: The MuJoCo MjData
        :param kd: Derivative gain (scalar or array). If scalar, apply the same gain to all actuators.
        :param ki: Integral gain for velocity error (helps overcome static friction)
        """
        
        self.model = model
        self.data = data

        # Map from actuators to DOF indices
        self.joint_ids = self.model.actuator_trnid[:, 0]   # Joint IDs controlled by actuators
        self.dof_indices = self.model.jnt_dofadr[self.joint_ids]  # DOF indices in qvel

        self.num_actuators = self.model.nu  # Number of actuators

        # Store KD as scalar or array
        if np.isscalar(kd):
            self.kd = np.full(self.num_actuators, kd, dtype=float)
        else:
            self.kd = np.array(kd, dtype=float)
            assert len(self.kd) == self.num_actuators, \
                   "kd array must match the number of actuators"

        # Store KI for integral term
        if np.isscalar(ki):
            self.ki = np.full(self.num_actuators, ki, dtype=float)
        else:
            self.ki = np.array(ki, dtype=float)
            assert len(self.ki) == self.num_actuators, \
                   "ki array must match the number of actuators"

        # Default velocity targets
        self.v_targets = np.zeros(self.num_actuators, dtype=float)
        
        # Integral of velocity error (for overcoming static friction)
        self.v_error_integral = np.zeros(self.num_actuators, dtype=float)
        
        # Maximum integral term to prevent windup
        self.integral_limit = 1.0
        
        # Add damping to reduce oscillations
        self.damping = 5.0
        
        # Static friction compensation
        self.static_friction_threshold = 0.005  # Velocity threshold for static friction
        self.static_friction_comp = 0.1
        
        # Trajectory function (optional)
        self.trajectory_function = None
        
        # For debugging
        self.debug = False
        
        # Previous velocity error for derivative term
        self.prev_v_error = np.zeros(self.num_actuators, dtype=float)
        
        # Low-pass filter for velocity measurements
        self.filtered_velocity = np.zeros(self.num_actuators, dtype=float)
        self.filter_coeff = 0.7  # Filter coefficient (0-1, higher = more filtering)

    def set_velocity_target(self, v_des):
        """
        Set a constant velocity target (if not using a trajectory).
        :param v_des: List or array of desired velocities, matching the number of actuators.
        """
        v_des = np.array(v_des, dtype=float)
        assert len(v_des) == self.num_actuators, \
               "v_des array must match the number of actuators"
        self.v_targets = v_des
        
        # Reset integral term when target changes significantly
        for i in range(self.num_actuators):
            if abs(v_des[i]) > 0.1 and abs(self.v_targets[i] - v_des[i]) > 0.1:
                self.v_error_integral[i] = 0.0

    def set_velocity_trajectory(self, trajectory_function):
        """
        Set a trajectory function to dynamically update velocity targets.
        :param trajectory_function: A function that takes time `t` as input and returns velocity targets.
        """
        self.trajectory_function = trajectory_function
        
        # Reset integral term when switching to trajectory
        self.v_error_integral = np.zeros(self.num_actuators, dtype=float)

    def control_callback(self, model, data):
        # If a trajectory function is provided, update v_targets based on time
        if self.trajectory_function:
            self.v_targets = self.trajectory_function(data.time)
        
        # Gravity compensation
        gravity_torques = data.qfrc_bias[self.dof_indices]
        data.ctrl[:] = gravity_torques

        # D-control torque with integral term and friction compensation
        for i in range(self.num_actuators):
            dof_i = self.dof_indices[i]
            v_actual = data.qvel[dof_i]
            
            # Apply low-pass filter to velocity measurements
            self.filtered_velocity[i] = self.filter_coeff * self.filtered_velocity[i] + \
                                       (1 - self.filter_coeff) * v_actual
            
            v_target = self.v_targets[i]
            
            # Velocity error using filtered velocity
            v_error = self.filtered_velocity[i] - v_target
            
            # Update integral term (with anti-windup)
            if abs(v_error) < 0.5:  # Only integrate when error is small
                self.v_error_integral[i] += v_error * model.opt.timestep
                self.v_error_integral[i] = np.clip(self.v_error_integral[i], 
                                                  -self.integral_limit, 
                                                  self.integral_limit)
            
            # Derivative term with damping
            torque_d = -self.kd[i] * v_error - self.damping * self.filtered_velocity[i]
            
            # Integral term
            torque_i = -self.ki[i] * self.v_error_integral[i]
            
            # Static friction compensation (only when velocity is near zero and target is non-zero)
            if abs(self.filtered_velocity[i]) < self.static_friction_threshold and abs(v_target) > 0:
                friction_comp = self.static_friction_comp * np.sign(v_target)
            else:
                friction_comp = 0
            
            # Apply all terms
            data.ctrl[i] += torque_d + torque_i + friction_comp
            
            # Store current error for next iteration
            self.prev_v_error[i] = v_error
            
            # Debug output
            if self.debug and i == 0 and data.time % 0.1 < model.opt.timestep:
                print(f"Joint {i}: v_target={v_target:.3f}, v_actual={v_actual:.3f}, "
                      f"filtered_v={self.filtered_velocity[i]:.3f}, "
                      f"error={v_error:.3f}, integral={self.v_error_integral[i]:.3f}, "
                      f"torque_d={torque_d:.3f}, torque_i={torque_i:.3f}, "
                      f"friction_comp={friction_comp:.3f}")

