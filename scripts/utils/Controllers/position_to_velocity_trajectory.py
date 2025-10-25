import mujoco
import numpy as np
import time

from utils.Controllers.Object_velocity_to_joint_velocity import VelocityMapper
from utils.Controllers.compute_object_admittance_control import ObjectAdmittanceController

class PositionTrajectory:
    def __init__(self, model, data, control_state, controller, kp_gains, max_velocities, position_thresholds, second_target_positions, actuators_per_robot, num_actuators, get_contact_wrenches, compute_object_admittance_control, object_velocity_to_joint_velocities):
        self.model = model
        self.data = data
        self.control_state = control_state
        self.controller = controller
        self.kp_gains = kp_gains
        self.max_velocities = max_velocities
        self.position_thresholds = position_thresholds
        self.second_target_positions = second_target_positions
        self.actuators_per_robot = actuators_per_robot
        self.num_actuators = num_actuators
        self.get_contact_wrenches = get_contact_wrenches
        self.compute_object_admittance_control = compute_object_admittance_control
        self.object_velocity_to_joint_velocities = object_velocity_to_joint_velocities

    def trajectory_callback(self, t):
        model = self.model
        data = self.data
        control_state = self.control_state
        num_actuators = self.num_actuators
        kp_gains = self.kp_gains
        max_velocities = self.max_velocities
        position_thresholds = self.position_thresholds
        second_target_positions = self.second_target_positions
        actuators_per_robot = self.actuators_per_robot

        joint_positions = np.zeros(num_actuators)
        for i in range(num_actuators):
            joint_id = model.actuator_trnid[i, 0]
            joint_positions[i] = data.qpos[model.jnt_qposadr[joint_id]]

        current_time = time.time()

        if current_time - control_state.get("last_print_time", 0) > 2.0:
            print(f"\nCurrent control phase: {control_state['phase']}")
            control_state["last_print_time"] = current_time

        if "active_control_strategy" not in control_state:
            control_state["active_control_strategy"] = "position_control"

        if control_state["phase"] in ["reaching_first_target", "reaching_second_target"]:
            control_state["active_control_strategy"] = "position_control"
            position_errors = control_state["current_targets"] - joint_positions
            velocity_commands = kp_gains * position_errors
            velocity_commands = np.clip(velocity_commands, -max_velocities, max_velocities)

            for i in range(num_actuators):
                if abs(position_errors[i]) < position_thresholds[i] * 0.5:
                    velocity_commands[i] = 0.0

            all_reached = np.all(np.abs(position_errors) <= position_thresholds)

            if all_reached:
                if control_state["phase"] == "reaching_first_target":
                    control_state["phase"] = "reaching_second_target"
                    control_state["current_targets"] = second_target_positions.copy()
                    control_state["phase_start_time"] = current_time
                    control_state["last_phase_change"] = current_time
                    print("\n🎯 Switching to second targets (grasp positions)...")
                    return np.zeros(num_actuators)
                elif control_state["phase"] == "reaching_second_target" and current_time - control_state["last_phase_change"] > 2.0:
                    control_state["phase"] = "reading_sensors"
                    control_state["last_phase_change"] = current_time
                    print("\n✅ All targets reached. Switching to sensor reading phase.")
                    return np.zeros(num_actuators)

            return velocity_commands

        elif control_state["phase"] == "reading_sensors":
            control_state["active_control_strategy"] = "none"
            if current_time - control_state["last_phase_change"] > 3.0:
                control_state["phase"] = "lifting_object"
                control_state["last_phase_change"] = current_time
                print("\n🔼 Starting object lifting phase...")
                print(f"Target lift height: {control_state['target_lift_height']:.4f} m")
            return np.zeros(num_actuators)

        elif control_state["phase"] == "lifting_object":
            print("[DEBUG] Entered lifting_object phase")
            current_time = time.time()
            box_body_id = control_state.get("box_body_id", -1)
            current_box_height = data.xpos[box_body_id][2] if box_body_id != -1 else 0.0

            lift_threshold = control_state["initial_box_height"] + 0.55 * (
                control_state["target_lift_height"] - control_state["initial_box_height"]
            )
            print(f"[DEBUG] Box height: {current_box_height:.3f}, Lift threshold: {lift_threshold:.3f}")
            print(f"[DEBUG] External force already applied? {control_state.get('external_force_applied', False)}")
            print(f"[DEBUG] box_body_id: {box_body_id}")

            if current_box_height >= lift_threshold and not control_state.get("external_force_applied", False):
                external_force = np.array([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])
                data.xfrc_applied[box_body_id] = external_force
                control_state["external_force_applied"] = True
                print(f"[DEBUG] External force applied array: {data.xfrc_applied[box_body_id]}")
                print(f"[DEBUG] Condition met? Height: {current_box_height} >= {lift_threshold}")
                control_state["external_force_time"] = current_time
                print(f"💥 External disturbance applied at height: {current_box_height:.3f} m")

            if control_state.get("external_force_applied", False):
                time_since_force = current_time - control_state.get("external_force_time", current_time)
                if time_since_force > 2.0:
                    data.xfrc_applied[box_body_id] = np.zeros(6)

            if current_box_height >= control_state["target_lift_height"] - 0.01:
                if not control_state.get("lifting_complete", False):
                    print(f"\n✅ TARGET HEIGHT REACHED! Current height: {current_box_height:.4f} m")
                    control_state["lifting_complete"] = True
                    control_state["phase"] = "maintaining_position"
                    control_state["last_phase_change"] = current_time
                    control_state["active_control_strategy"] = "none"
                return np.zeros(num_actuators)

            left_wrench, right_wrench, contact_detected = self.get_contact_wrenches(data)

            if contact_detected:
                object_velocity = self.compute_object_admittance_control(
                    data=data,
                    left_wrench=left_wrench,
                    right_wrench=right_wrench,
                    desired_height=control_state["target_lift_height"]
                )

                # OPTIONAL: reduce lateral noise
                object_velocity[:2][np.abs(object_velocity[:2]) < 0.005] = 0.0

                joint_velocities = self.object_velocity_to_joint_velocities(data, object_velocity)
                control_state["active_control_strategy"] = "admittance_control"
                print(f"🔄 ACTIVE CONTROL: ADMITTANCE CONTROL")
                return joint_velocities
            else:
                control_state["active_control_strategy"] = "waiting_for_contact"
                print(f"🔄 ACTIVE CONTROL: WAITING FOR CONTACT")
                return np.zeros(num_actuators)

        elif control_state["phase"] == "maintaining_position":
            control_state["active_control_strategy"] = "position_maintenance"
            return np.zeros(num_actuators)

        else:
            control_state["active_control_strategy"] = "unknown"
            print(f"Unknown control phase: {control_state['phase']}")
            return np.zeros(num_actuators)
