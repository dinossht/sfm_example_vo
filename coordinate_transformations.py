import numpy as np


def R_cam_imu_matrix():
    # IMU origo/pos given in camera coordinate
    imuOrigo_cam = np.array([0.71, 1.34, -3.53])
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

    print("Array x vector given in IMU, this can be ferries forward acceleration:\t", end="")
    xVec_imu = np.array([3.60000, 0.10000, -1.3400])  # camera origo given in imu
    print(xVec_imu)

    R_cam_imu = R_zxy_xyz @ R_z(-13)
    print("Same vector given in camera: \t\t\t\t\t\t", end="")
    print(R_cam_imu @ xVec_imu)
    print("Use this to rotate IMU acc data and euler angles")

    # RTK origo given in IMU 
    rtkOrigo_imu = np.array([-0.025, 0.1, 0.06])
    print("Assume that they have same origo, too small offset")

    return R_cam_imu, rtkOrigo_imu, R_zxy_xyz