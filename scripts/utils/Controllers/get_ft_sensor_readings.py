import mujoco

class FtSensorReader:
    def __init__(self, model):
        self.model = model

    def get_ft_sensor_readings(self, data):
        """Get force-torque sensor readings from the model."""
        sensor_readings = {}
        
        # Iterate through all sensors
        for i in range(self.model.nsensor):
            sensor_name = self.model.sensor(i).name
            sensor_type = self.model.sensor(i).type
            
            # Check if it's a force or torque sensor
            if sensor_type == mujoco.mjtSensor.mjSENS_FORCE or sensor_type == mujoco.mjtSensor.mjSENS_TORQUE:
                # Get the sensor data
                adr = int(self.model.sensor(i).adr)  # Convert to int to avoid indexing error
                dim = int(self.model.sensor(i).dim)  # Convert to int to avoid indexing error
                sensor_readings[sensor_name] = data.sensordata[adr:adr+dim].copy()
        
        # Print sensor readings
        if sensor_readings:
            print("\n--- FT SENSOR READINGS ---")
            for name, values in sensor_readings.items():
                print(f"{name}: {values}")
        
        return sensor_readings