from scipy.spatial.transform import Rotation as R
import numpy as np

class PosRot:
    def quaternion_multiply(self, q1, q2):
        # Extract quaternion components from Rotation objects
        q1 = q1.as_quat()  # [x1, y1, z1, w1]
        q2 = q2.as_quat()  # [x2, y2, z2, w2]
        
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        # Perform quaternion multiplication
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        # Return the result as a new Rotation object
        return R.from_quat([x, y, z, w])

    def quaternion_inverse(self, q):
        # Extract quaternion components as an array [x, y, z, w]
        quat = q.as_quat()  # Extract [x, y, z, w] from Rotation object
        x, y, z, w = quat  # Unpack the quaternion
        # Return the inverse of the quaternion
        return R.from_quat([-x, -y, -z, w])

class PoseErrorCalculator:
    def __init__(self):
        """Initialize the PoseErrorCalculator."""
        self.posrot = PosRot()

    def compute_error(self, pose_des, pose_act):
        """
        Compute the error vector between two poses.

        Args:
            pose_des (np.ndarray): Desired pose [x, y, z, qw, qx, qy, qz].
            pose_act (np.ndarray): Actual pose [x, y, z, qw, qx, qy, qz].

        Returns:
            np.ndarray: Error vector [dx, dy, dz, roll_error, pitch_error, yaw_error].
        """
        # Extract position and quaternion components
        pos_des, quat_des = pose_des[:3], pose_des[3:]  # Desired position and orientation
        pos_act, quat_act = pose_act[:3], pose_act[3:]  # Actual position and orientation

        # Compute position error as (desired - actual)
        pos_error = pos_des - pos_act

        # Convert to Rotation objects
        # Remember: pose given as [qw, qx, qy, qz]
        # R.from_quat expects [x, y, z, w]
        rot_des = R.from_quat([quat_des[1], quat_des[2], quat_des[3], quat_des[0]])  # Desired orientation
        rot_act = R.from_quat([quat_act[1], quat_act[2], quat_act[3], quat_act[0]])  # Actual orientation

        # Compute relative rotation as q_rel = q_des * q_act^(-1)
        rot_act_inv = self.posrot.quaternion_inverse(rot_act)
        rot_relative = self.posrot.quaternion_multiply(rot_des, rot_act_inv)

        # Convert the relative rotation to Euler angles [roll, pitch, yaw]
        euler_error = rot_relative.as_euler('xyz', degrees=False)

        # Combine position and orientation errors into a single vector
        error_vector = np.hstack((pos_error, euler_error))

        return error_vector
