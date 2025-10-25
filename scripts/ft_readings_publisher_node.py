#!/usr/bin/env python
import rospy
import serial
import struct
from geometry_msgs.msg import WrenchStamped

# FT sensor configuration
PORT = '/dev/ttyUSB0'
BAUD = 115200
DF = 50.0     # Force divisor
DT = 1000.0   # Torque divisor

def compute_checksum(data_bytes):
    return sum(data_bytes) & 0xFF

def parse_force_torque(data):
    # Extract 6 values (3 force, 3 torque) from data bytes
    ft_raw = [struct.unpack('>h', bytes(data[i:i+2]))[0] for i in range(1, 13, 2)]
    force = [ft_raw[i] / DF for i in range(3)]
    torque = [ft_raw[i] / DT for i in range(3, 6)]
    return force, torque

def main():
    rospy.init_node('robotous_ft_publisher')
    pub = rospy.Publisher('/ft_sensor', WrenchStamped, queue_size=10)

    try:
        ser = serial.Serial(PORT, BAUD, timeout=None)
        ser.reset_input_buffer()

        # Send start command
        command = [0x0B] + [0x00] * 7
        checksum = compute_checksum(command)
        packet = bytes([0x55] + command + [checksum, 0xAA])
        ser.write(packet)
        rospy.loginfo("Sent start command to FT sensor.")

        rate = rospy.Rate(100)  # 100 Hz

        while not rospy.is_shutdown():
            if ser.read() != b'\x55':
                continue

            data = ser.read(16)
            if len(data) != 16:
                continue

            received_checksum = ser.read(1)
            if len(received_checksum) != 1 or ser.read() != b'\xAA':
                continue

            if compute_checksum(data) != ord(received_checksum):
                rospy.logwarn("Checksum mismatch")
                continue

            force, torque = parse_force_torque(data)

            msg = WrenchStamped()
            msg.header.stamp = rospy.Time.now()
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = force
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = torque
            pub.publish(msg)

            # Optional debug print
            rospy.loginfo_throttle(1, "Force: {:.2f} {:.2f} {:.2f} | Torque: {:.3f} {:.3f} {:.3f}".format(
                *force, *torque))

            rate.sleep()

    except serial.SerialException as e:
        rospy.logerr("Serial exception: {}".format(e))
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
