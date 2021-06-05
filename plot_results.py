import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np


def plot_xy_data(data, ref_data, xlabel="y (m)", ylabel="x (m)"):
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    plt.plot(ref_data[:, 1], ref_data[:, 0], color="black", linestyle="dashed", label="reference")
    plt.plot(data[:, 1], ref_data[:, 0], color="blue", alpha=0.7)

    plt.xlabel(ylabel)
    plt.ylabel(xlabel)

    plt.legend(loc="upper left")
    plt.grid()

    plt.show()


def plot_xy_data_w_error(data, ref_data, xlabel="y (m)", ylabel="x (m)"):
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

    ape = np.linalg.norm(data - ref_data, ord=1, axis=1)
    vmin = min(ape)
    vmax = max(ape)
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    plt.plot(ref_data[:, 1], ref_data[:, 0], color="black", linestyle="dashed", label="reference")
    #plt.plot(data[:, 1], data[:, 0], color="blue", alpha=0.7)
    im = ax.scatter(data[:, 1], data[:, 0], marker=".", c=cmap(norm(ape)))

    plt.legend(loc="upper left")
    plt.grid()
    plt.title("Error mapped onto trajectory")

    plt.xlabel(ylabel)
    plt.ylabel(xlabel)
    plt.show()

def plot_xyz(data, ref_data, timestamp):
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    plt.subplot(311)
    plt.plot(ref_data[:, 0], color="black", linestyle="dashed", label="reference")
    plt.plot(ref_data[:, 0], color="blue", alpha=0.7)
    plt.ylabel("x (m)")
    plt.legend()

    plt.subplot(312)
    plt.plot(ref_data[:, 1], color="black", linestyle="dashed", label="reference")
    plt.plot(ref_data[:, 1], color="blue", alpha=0.7)
    plt.ylabel("y (m)")

    plt.subplot(313)
    plt.plot(ref_data[:, 2], color="black", linestyle="dashed", label="reference")
    plt.plot(ref_data[:, 2], color="blue", alpha=0.7)
    plt.ylabel("z (m)")

    plt.xlabel("t (s)")
    plt.show()

def plot_rpy(data, ref_data):
    pass

def plot_xy_ape(data, ref_data):
    pass

def plot_stats_summary():
    pass


from load_ros_camera_rtk import camRtkData

trip_nr = 4  # pass på sensorer er montert (pos og heading) ulike og vil på virke det for ulike dager

dat = camRtkData(720, trip_nr)
IMU_data, IMU_times = dat.get_imu_in_body()
GNSS2_data, GNSS2_times, equiv_rtk_data = dat.get_gnss2_pose_in_ned()

#plot_xy_data(GNSS2_data, equiv_rtk_data)
plot_xy_data_w_error(GNSS2_data, equiv_rtk_data)