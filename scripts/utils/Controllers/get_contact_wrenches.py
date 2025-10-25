import mujoco
import numpy as np

class ContactWrenchReader:
    def __init__(self, model):
        self.model = model

    def get_contact_wrenches(self, data):
        """
        Get wrenches (force/torque) at contact points between end-effectors and box.
        Prints the wrench values when contact is detected.
        """
        # Find box body ID
        box_body_id = -1
        for i in range(self.model.nbody):
            if "box" in self.model.body(i).name:
                box_body_id = i
                break
        
        if box_body_id == -1:
            return np.zeros(6), np.zeros(6), False
        
        # Get all geom IDs for the box
        box_geom_ids = []
        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] == box_body_id:
                box_geom_ids.append(i)
        
        # Initialize wrenches
        left_wrench = np.zeros(6)
        right_wrench = np.zeros(6)
        
        # Check all contacts
        contact_detected = False
        for i in range(data.ncon):
            contact = data.contact[i]
            
            # Check if this contact involves the box
            if contact.geom1 in box_geom_ids or contact.geom2 in box_geom_ids:
                # Get contact force
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, data, i, contact_force)
                
                # Determine which end effector is involved
                other_geom = contact.geom2 if contact.geom1 in box_geom_ids else contact.geom1
                other_body = self.model.geom_bodyid[other_geom]
                body_name = self.model.body(other_body).name
                
                # Compute torque around box center
                contact_pos = contact.pos.copy()
                box_pos = data.xpos[box_body_id].copy()
                lever_arm = contact_pos - box_pos
                torque = np.cross(lever_arm, contact_force[:3])
                
                # Add to appropriate wrench
                if "left" in body_name.lower():
                    left_wrench[:3] += contact_force[:3]
                    left_wrench[3:] += torque
                    contact_detected = True
                elif "right" in body_name.lower():
                    right_wrench[:3] += contact_force[:3]
                    right_wrench[3:] += torque
                    contact_detected = True
        
        # Print wrenches if contact is detected
        if contact_detected:
            print("\n--- CONTACT WRENCHES ---")
            print(f"Left end-effector wrench:")
            print(f"  Force: [{left_wrench[0]:.4f}, {left_wrench[1]:.4f}, {left_wrench[2]:.4f}]N")
            print(f"  Torque: [{left_wrench[3]:.4f}, {left_wrench[4]:.4f}, {left_wrench[5]:.4f}]Nm")
            print(f"  Force magnitude: {np.linalg.norm(left_wrench[:3]):.4f}N")
            
            print(f"\nRight end-effector wrench:")
            print(f"  Force: [{right_wrench[0]:.4f}, {right_wrench[1]:.4f}, {right_wrench[2]:.4f}]N")
            print(f"  Torque: [{right_wrench[3]:.4f}, {right_wrench[4]:.4f}, {right_wrench[5]:.4f}]Nm")
            print(f"  Force magnitude: {np.linalg.norm(right_wrench[:3]):.4f}N")
            
            # Calculate combined wrench
            combined_wrench = left_wrench + right_wrench
            print(f"\nCombined wrench:")
            print(f"  Force: [{combined_wrench[0]:.4f}, {combined_wrench[1]:.4f}, {combined_wrench[2]:.4f}]N")
            print(f"  Torque: [{combined_wrench[3]:.4f}, {combined_wrench[4]:.4f}, {combined_wrench[5]:.4f}]Nm")
            print(f"  Force magnitude: {np.linalg.norm(combined_wrench[:3]):.4f}N")
            
            # Get object mass and compute weight
            object_mass = self.model.body_mass[box_body_id]
            weight_force = object_mass * 9.81  # N
            print(f"\nObject mass: {object_mass:.3f}kg, Weight: {weight_force:.3f}N")
            
            # Check if lifting force exceeds weight
            if combined_wrench[2] > weight_force:
                print("✅ Lifting force exceeds object weight")
            else:
                print("⚠️ Lifting force insufficient to overcome weight")
        
        return left_wrench, right_wrench, contact_detected