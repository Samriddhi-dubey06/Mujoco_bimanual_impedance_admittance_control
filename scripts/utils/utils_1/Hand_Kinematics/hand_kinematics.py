import numpy as np

class HandKinematics:
    """
    A class to compute the hand Jacobian and grasp sub-matrix for robotic manipulators.

    Methods:
        grasp_sub_matrix_calculator: Computes the grasp sub-matrix G_i for a single contact point.
        hand_jacobian_calculator: Computes the new hand Jacobian matrix (12x12).
    """

    @staticmethod
    def grasp_sub_matrix_calculator(rotation_matrix_input, si, ti, ni, bi):
        """
        Calculate the grasp sub-matrix G_i corresponding to a single contact point.

        Args:
            rotation_matrix_input (np.ndarray): (3x3) Rotation matrix from object frame to contact frame.
            si, ti, ni (np.ndarray): (3x1) Vectors forming an orthonormal basis for the contact frame.
            bi (np.ndarray): (3x1) Vector representing the contact point relative to the object center of mass.

        Returns:
            np.ndarray: (6x6) Grasp sub-matrix G_i.
        """
        G_i = np.zeros((6, 6))
        G_i[0:3, 0] = si
        G_i[0:3, 1] = ti
        G_i[0:3, 2] = ni
        G_i[3:, 0] = np.cross(np.dot(rotation_matrix_input, bi), si)
        G_i[3:, 1] = np.cross(np.dot(rotation_matrix_input, bi), ti)
        G_i[3:, 2] = np.cross(np.dot(rotation_matrix_input, bi), ni)
        G_i[3:, 3] = si
        G_i[3:, 4] = ti
        G_i[3:, 5] = ni
        return G_i
    
    @staticmethod
    def point_contact_grasp_sub_matrix_calculator(rotation_matrix_input, si, ti, ni, bi):
        """
        Calculate the grasp sub-matrix G_i corresponding to a single contact point.

        Args:
            rotation_matrix_input (np.ndarray): (3x3) Rotation matrix from object frame to contact frame.
            si, ti, ni (np.ndarray): (3x1) Vectors forming an orthonormal basis for the contact frame.
            bi (np.ndarray): (3x1) Vector representing the contact point relative to the object center of mass.

        Returns:
            np.ndarray: (6x6) Grasp sub-matrix G_i.
        """
        G_i = np.zeros((6, 3))
        G_i[0:3, 0] = si
        G_i[0:3, 1] = ti
        G_i[0:3, 2] = ni
        G_i[3:, 0] = np.cross(np.dot(rotation_matrix_input, bi), si)
        G_i[3:, 1] = np.cross(np.dot(rotation_matrix_input, bi), ti)
        G_i[3:, 2] = np.cross(np.dot(rotation_matrix_input, bi), ni)
        return G_i

    @staticmethod
    def hand_jacobian_calculator(Wpki_list, Rpki_list, manipulator_full_jacobian_list):
        """
        Calculate the new hand Jacobian matrix, size 12x12.

        Args:
            Wpki_list (list of np.ndarray): List of (3x3) matrices defining contact frame bases for each contact.
            Rpki_list (list of np.ndarray): List of (3x3) rotation matrices for each contact frame.
            manipulator_full_jacobian_list (list of np.ndarray): List of full Jacobians for each manipulator,
                                                                where each Jacobian is (6xDOF).

        Returns:
            np.ndarray: (12x12) New hand Jacobian matrix.
        """
        num_fingers = len(manipulator_full_jacobian_list)  # Number of fingers or manipulators
        dof = manipulator_full_jacobian_list[0].shape[1]  # Degrees of freedom for each manipulator
        Jh_new = np.zeros((12, 12))  # Initialize the hand Jacobian matrix (12x12)

        for i in range(num_fingers):
            Wpki = Wpki_list[i]  # Contact frame basis for the i-th contact (3x3)
            Rpki = Rpki_list[i]  # Rotation matrix for the i-th contact (3x3)
            Ji = manipulator_full_jacobian_list[i]  # Full Jacobian of the i-th manipulator (6xDOF)

            # Create the block diagonal transformation matrices
            W_block = np.block([
                [Wpki.T, np.zeros_like(Wpki.T)],  # Block diagonal for Wpki
                [np.zeros_like(Wpki.T), Wpki.T]  # Second diagonal for Wpki
            ])  # Resulting matrix: (6x6)

            R_block = np.block([
                [Rpki, np.zeros_like(Rpki)],  # Block diagonal for Rpki
                [np.zeros_like(Rpki), Rpki]  # Second diagonal for Rpki
            ])  # Resulting matrix: (6x6)

            # Transform and store the new Jacobian block
            Jh_new[6 * i:6 * i + 6, dof * i:dof * i + dof] = np.dot(W_block, np.dot(R_block, Ji))

        return Jh_new

    @staticmethod
    def hand_jacobian_calculator_positional(Wpki_list, Rpki_list, manipulator_full_jacobian_list):
        """
        Calculate the hand Jacobian matrix for positional components only, size 6x12.

        Args:
            Wpki_list (list of np.ndarray): List of (3x3) matrices defining contact frame bases for each contact.
            Rpki_list (list of np.ndarray): List of (3x3) rotation matrices for each contact frame.
            manipulator_full_jacobian_list (list of np.ndarray): List of full Jacobians for each manipulator,
                                                                where each Jacobian is (6xDOF).

        Returns:
            np.ndarray: (6x12) Hand Jacobian matrix for positional components.
        """
        num_fingers = len(manipulator_full_jacobian_list)  # Number of fingers or manipulators
        dof = manipulator_full_jacobian_list[0].shape[1]  # Degrees of freedom for each manipulator
        Jh_positional = np.zeros((6, 12))  # Initialize the hand Jacobian matrix (6x12)

        for i in range(num_fingers):
            Wpki = Wpki_list[i]  # Contact frame basis for the i-th contact (3x3)
            Rpki = Rpki_list[i]  # Rotation matrix for the i-th contact (3x3)
            Ji = manipulator_full_jacobian_list[i]  # Full Jacobian of the i-th manipulator (6xDOF)

            # Only consider positional components (first 3 rows of Ji)
            Ji_positional = Ji[:3, :]  # Extract the positional part of the Jacobian (3xDOF)

            # Transform and store the new Jacobian block
            Jh_positional[3 * i:3 * i + 3, dof * i:dof * i + dof] = np.dot(Wpki.T, np.dot(Rpki, Ji_positional))

        return Jh_positional
