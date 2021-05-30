import numpy as np


# IMU origo/pos given in camera coordinate
imuOrigo_cam = np.array([0.42, 1.34, -3.58])
print("See geogebra sketch\n")

# Rotation matrix to rotation IMU to camera
# First rotate yaw of 8.303 degrees
def R_z(yaw_deg):
    y = yaw_deg * np.pi / 180
    return np.array([
    [np.cos(y),     -np.sin(y), 0],
    [np.sin(y),     np.cos(y),  0],
    [0,             0,          1]])

R_zxy_xyz = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]])

print("Array x vector given in IMU")
xVec_imu = np.array([1, 0, 0])
print(xVec_imu)

R_cam_imu = R_zxy_xyz @ R_z(-8.303)
print("Same vector given in camera")
print(R_cam_imu @ xVec_imu)

# RTK origo given in IMU 
rtkOrigo_imu = np.array([-0.025, 0.1, 0.06])
print("Assume that they have same origo, too small offset")