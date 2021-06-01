from sensor_dataset import ROSDataset
from ground_truth import ROSGroundTruth
import matplotlib.pyplot as plt
from coordinate_transformations import R_cam_imu_matrix
import numpy as np


def plot_trajectory(fignum, poses, scale=1, title="Plot Trajectory", axis_labels=('X axis', 'Y axis', 'Z axis')):
    """
    Plot a complete 2D/3D trajectory using poses in `values`.

    Args:
        fignum (int): Integer representing the figure number to use for plotting.
        values (gtsam.Values): Values containing some Pose2 and/or Pose3 values.
        scale (float): Value to scale the poses by.
        marginals (gtsam.Marginals): Marginalized probability values of the estimation.
            Used to plot uncertainty bounds.
        title (string): The title of the plot.
        axis_labels (iterable[string]): List of axis labels to set.
    """
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    axes.set_zlabel(axis_labels[2])
    axes.view_init(elev=0, azim=-90)

    # Then 3D poses, if any
    for pose in poses:
        plot_pose3_on_axes(axes, pose, axis_length=scale)

    fig.suptitle(title)
    fig.canvas.set_window_title(title.lower())


def plot_pose3_on_axes(axes, pose, axis_length=0.1, scale=1):
    """
    Plot a 3D pose on given axis `axes` with given `axis_length`.

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        point (gtsam.Point3): The point to be plotted.
        linespec (string): String representing formatting options for Matplotlib.
        P (numpy.ndarray): Marginal covariance matrix to plot the uncertainty of the estimation.
    """
    # get rotation and translation (center)
    gRp = pose[:3,:3]  # rotation from pose to global
    origin = pose[:3,3]

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin[np.newaxis], z_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')


def set_axes_equal(fignum):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
      fignum (int): An integer representing the figure number for Matplotlib.
    """
    fig = plt.figure(fignum)
    ax = fig.gca(projection='3d')

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


class camRtkData:
    def __init__(self, t_start):
        self.init = False

        trip_nr = 3
        self.dat = ROSDataset("dataset/rosbags", f"trondheim{trip_nr}_inn", t_start, -1)
        self.gt = ROSGroundTruth(f"dataset/groundtruths/obsv_estimates ({trip_nr}).mat", "dataset/groundtruths/ned_origin.mat", trip_nr)


    def get_rtk_pos_in_IMU(self, timestamp):
        if not self.init:
            self.init = True

            T0 = self.gt.get_T(timestamp)
            # NED to body
            T0_body_ned = np.linalg.inv(T0)
            R_cam_imu, rtkOrigo_imu, R_zxy_xyz = R_cam_imu_matrix()

            Trans = np.eye(4)
            # Convert to imu frame
            Trans[:3,:3] = R_cam_imu
            Trans = Trans @ T0_body_ned
            # Move to IMU origo
            #Trans[:3,3] += rtkOrigo_imu
            self.T_matrix = Trans

        T = self.T_matrix @ self.gt.get_T(timestamp)
        return T

    def get_img_rtk_pos_in_IMU(self):
        img, _, timestamp = self.dat.get_image()
        T = self.get_rtk_pos_in_IMU(timestamp)
        return img, T

"""
dat = camRtkData()

poses = []
for _ in range(20):#510
    img, T = dat.get_img_rtk_pos_in_IMU()
    poses.append(T)
    
plot_trajectory(1, poses)
set_axes_equal(1)        
plt.show()
"""



