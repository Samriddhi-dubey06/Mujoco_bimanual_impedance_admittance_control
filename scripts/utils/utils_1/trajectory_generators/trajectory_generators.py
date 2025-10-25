from scipy.interpolate import CubicSpline
import numpy as np
# Replace the tf import with transforms3d
import transforms3d
# Define a compatibility layer for tf.transformations
class TFCompatibility:
    @staticmethod
    def quaternion_slerp(q1, q2, fraction):
        """Spherical linear interpolation between two quaternions."""
        return transforms3d.quaternions.qslerp(q1, q2, fraction)

tf = TFCompatibility()


class TrajectoryGenerator:
    def __init__(self, total_time=15.0, time_step=0.001):
        """
        Initialize the TrajectoryGenerator.

        Args:
            total_time (float): Total time for the trajectory (seconds).
            time_step (float): Time step for the trajectory generation (seconds).
        """
        self.total_time = total_time
        self.time_step = time_step

    def generate_position_trajectory(self, start_pos, end_pos):
        """
        Generate a smooth position trajectory using cubic spline interpolation.

        Args:
            start_pos (list or np.ndarray): Starting position [x, y, z].
            end_pos (list or np.ndarray): Ending position [x, y, z].

        Returns:
            tuple: A tuple containing:
                - position_trajectory (list of list): Interpolated position trajectory [[x, y, z], ...].
                - t_values (np.ndarray): Time values corresponding to the trajectory.
        """
        points = [start_pos, end_pos]
        t_points = np.linspace(0, self.total_time, len(points))

        # Separate x, y, z components
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        z_points = [p[2] for p in points]

        # Create cubic splines for each component
        x_spline = CubicSpline(t_points, x_points)
        y_spline = CubicSpline(t_points, y_points)
        z_spline = CubicSpline(t_points, z_points)

        # Create the time steps
        t_values = np.arange(0, self.total_time, self.time_step)

        # Generate the smooth trajectory
        position_trajectory = [[x_spline(t), y_spline(t), z_spline(t)] for t in t_values]

        return position_trajectory, t_values

    def generate_orientation_trajectory(self, start_quat, end_quat, t_values):
        """
        Generate a smooth orientation trajectory using SLERP.

        Args:
            start_quat (list or np.ndarray): Starting orientation quaternion [qx, qy, qz, qw].
            end_quat (list or np.ndarray): Ending orientation quaternion [qx, qy, qz, qw].
            t_values (np.ndarray): Time values for the trajectory.

        Returns:
            list of list: Interpolated orientation trajectory [[qx, qy, qz, qw], ...].
        """
        orientation_trajectory = []

        # Perform SLERP for the entire trajectory
        for t in t_values:
            interpolated_orientation = tf.quaternion_slerp(start_quat, end_quat, t / self.total_time)
            orientation_trajectory.append(interpolated_orientation)

        return orientation_trajectory

    def generate_full_trajectory(self, start_pose, end_pose):
        """
        Generate a complete trajectory (position + orientation) for a box.

        Args:
            start_pose (list): Starting pose [x, y, z, qx, qy, qz, qw].
            end_pose (list): Ending pose [x, y, z, qx, qy, qz, qw].

        Returns:
            dict: A dictionary containing:
                - 'position': Position trajectory [[x, y, z], ...].
                - 'orientation': Orientation trajectory [[qx, qy, qz, qw], ...].
                - 'time': Time values corresponding to the trajectory.
        """
        # Extract position and orientation
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        end_pos, end_quat = end_pose[:3], end_pose[3:]

        # Generate position trajectory
        position_trajectory, t_values = self.generate_position_trajectory(start_pos, end_pos)

        # Generate orientation trajectory
        orientation_trajectory = self.generate_orientation_trajectory(start_quat, end_quat, t_values)

        return {
            'position': position_trajectory,
            'orientation': orientation_trajectory,
            'time': t_values
        }
