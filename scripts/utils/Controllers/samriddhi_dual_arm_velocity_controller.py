import mujoco
import numpy as np

class VelocityControllerGC:
    """
    A velocity controller for MuJoCo robots with gravity compensation.
    Based on the working example with additional filtering and control terms.
    """
    def __init__(self, model, data, kd=None, ki=0.05):
        """
        Initialize the controller.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            kd: Derivative gain (array of size num_actuators or scalar)
            ki: Integral gain (array of size num_actuators or scalar)
        """
        self.model = model
        self.data = data
        self.num_actuators = model.nu
        
        # Set default gains if not provided
        if kd is None:
            self.kd = np.ones(self.num_actuators) * 100.0
        elif np.isscalar(kd):
            self.kd = np.ones(self.num_actuators) * kd
        else:
            self.kd = kd
            
        # Set integral gains
        if np.isscalar(ki):
            self.ki = np.ones(self.num_actuators) * ki
        else:
            self.ki = ki
            
        # Default velocity targets
        self.v_targets = np.zeros(self.num_actuators, dtype=float)
        
        # Integral of velocity error (for overcoming static friction)
        self.v_error_integral = np.zeros(self.num_actuators, dtype=float)
        
        # Maximum integral term to prevent windup
        self.integral_limit = 0.2
        
        # Trajectory function (optional)
        self.trajectory_function = None
        
        # For debugging
        self.debug = False
        
        # Filtered velocity (for noise reduction)
        self.filtered_velocity = np.zeros(self.num_actuators, dtype=float)
        self.filter_coeff = 0.7
        
        # Store DOF indices for efficiency
        joint_ids = model.actuator_trnid[:, 0]
        self.dof_indices = model.jnt_dofadr[joint_ids]
        
    def set_velocity_target(self, v_targets):
        """Set target velocities for all actuators."""
        if len(v_targets) != self.num_actuators:
            raise ValueError(f"Expected {self.num_actuators} velocity targets, got {len(v_targets)}")
        self.v_targets = np.array(v_targets)
        
    def set_velocity_trajectory(self, trajectory_function):
        """Set a time-varying velocity trajectory function."""
        self.trajectory_function = trajectory_function
        
    def control_callback(self, model, data):
        """MuJoCo control callback function with additional filtering and control terms."""
        # If a trajectory function is provided, update v_targets based on time
        if self.trajectory_function:
            try:
                new_targets = self.trajectory_function(data.time)
                if new_targets is not None:
                    self.v_targets = new_targets
                elif self.v_targets is None:
                    # Initialize with zeros if targets are None
                    self.v_targets = np.zeros(self.num_actuators, dtype=float)
            except Exception as e:
                print(f"Error in trajectory function: {e}")
                # Ensure v_targets is never None
                if self.v_targets is None:
                    self.v_targets = np.zeros(self.num_actuators, dtype=float)
        elif self.v_targets is None:
            # Initialize with zeros if no trajectory function and targets are None
            self.v_targets = np.zeros(self.num_actuators, dtype=float)
        
        # Gravity compensation
        joint_ids = model.actuator_trnid[:, 0]
        dof_indices = model.jnt_dofadr[joint_ids]
        gravity_torques = data.qfrc_bias[dof_indices]
        data.ctrl[:] = gravity_torques

        # Control loop for each actuator
        for i in range(self.num_actuators):
            dof_i = dof_indices[i]
            v_actual = data.qvel[dof_i]
            
            # Apply low-pass filter to velocity measurements
            self.filtered_velocity[i] = self.filter_coeff * self.filtered_velocity[i] + \
                                       (1 - self.filter_coeff) * v_actual
            
            v_target = self.v_targets[i]
            
            # Use filtered velocity for error calculation
            v_error = self.filtered_velocity[i] - v_target
            
            # Update integral term (with anti-windup)
            if abs(v_error) < 0.1:  # Only integrate when error is small
                self.v_error_integral[i] += v_error * model.opt.timestep
                self.v_error_integral[i] = np.clip(self.v_error_integral[i], 
                                                  -self.integral_limit, 
                                                  self.integral_limit)
            else:
                # Decay integral when error is large
                self.v_error_integral[i] *= 0.95
            
            # D-control torque (like in the working example)
            torque_d = -self.kd[i] * v_error
            
            # I-control torque (small)
            torque_i = -self.ki[i] * self.v_error_integral[i]
            
            # Apply both terms
            data.ctrl[i] += torque_d + torque_i
            
            # Add extra torque for high-speed movements
            if abs(v_target) > 0.5:  # If we're commanding a high velocity
                # Add extra torque in the direction of motion to overcome friction
                extra_torque = np.sign(v_target) * 0.5
                data.ctrl[i] += extra_torque
            
            # Debug output
            if self.debug and i == 0 and data.time % 0.5 < model.opt.timestep:
                print(f"Joint {i}: v_target={v_target:.3f}, v_actual={v_actual:.3f}, "
                      f"filtered_v={self.filtered_velocity[i]:.3f}, error={v_error:.3f}, "
                      f"integral={self.v_error_integral[i]:.3f}, torque_d={torque_d:.3f}, "
                      f"torque_i={torque_i:.3f}, gravity={gravity_torques[i]:.3f}, "
                      f"total={data.ctrl[i]:.3f}")

