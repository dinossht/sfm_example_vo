from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.font_manager
import numpy as np
import datetime
import shutil
import os


def plot_xy_data(data, ref_data, xlabel="y (m)", ylabel="x (m)", savename=None):
    plt.cla()
    plt.clf()
    plt.close()
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    plt.plot(ref_data[:,1,3], ref_data[:,0,3], color="black", linestyle="dashed", label="reference")
    plt.plot(data[:,1,3], ref_data[:,0,3], color="blue", alpha=0.7)

    plt.xlabel(ylabel)
    plt.ylabel(xlabel)

    plt.legend(loc="upper left")
    plt.grid()

    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')

def plot_xy_data_w_error(data, ref_data, xlabel="y (m)", ylabel="x (m)", savename=None):
    plt.cla()
    plt.clf()
    plt.close()
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

    ate = np.linalg.norm(data[:,:2,3] - ref_data[:,:2,3], ord=1, axis=1)
    vmin = min(ate)
    vmax = max(ate)
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    plt.plot(ref_data[:,1,3], ref_data[:,0,3], color="black", linestyle="dashed", label="reference")
    im = ax.scatter(data[:,1,3], data[:,0,3], marker=".", c=cmap(norm(ate)))

    plt.legend(loc="upper left")
    plt.grid()
    plt.title("Error mapped onto trajectory")

    plt.xlabel(ylabel)
    plt.ylabel(xlabel)

    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')

def plot_xyz(data, ref_data, timestamp, savename=None):
    plt.cla()
    plt.clf()
    plt.close()
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
    plt.subplot(311)
    plt.plot(timestamp, ref_data[:,0,3], color="black", linestyle="dashed", label="reference")
    plt.plot(timestamp, data[:,0,3], color="blue", alpha=0.7)
    plt.ylabel("x (m)")
    plt.legend(loc="upper left")
    plt.xlim([min(timestamp), max(timestamp)])
    plt.grid()

    plt.subplot(312)
    plt.plot(timestamp, ref_data[:,1,3], color="black", linestyle="dashed", label="reference")
    plt.plot(timestamp, data[:,1,3], color="blue", alpha=0.7)
    plt.xlim([min(timestamp), max(timestamp)])
    plt.ylabel("y (m)")
    plt.grid()

    plt.subplot(313)
    plt.plot(timestamp, ref_data[:,2,3], color="black", linestyle="dashed", label="reference")
    plt.plot(timestamp, data[:,2,3], color="blue", alpha=0.7)
    plt.xlim([min(timestamp), max(timestamp)])
    plt.ylabel("z (m)")
    plt.grid()

    plt.xlabel("t (s)")
    plt.tight_layout()

    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')

def plot_rpy(data, ref_data, timestamp, savename=None):
    plt.cla()
    plt.clf()
    plt.close()
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

    eulers, eulers_ref = [], []
    for d, r in zip(data, ref_data):
        [yaw, pitch, roll] = R.from_matrix(d[:3,:3]).as_euler("zyx", degrees=True)
        [yaw_ref, pitch_ref, roll_ref] = R.from_matrix(r[:3,:3]).as_euler("zyx", degrees=True)
        eulers.append([roll, pitch, yaw])
        eulers_ref.append([roll_ref, pitch_ref, yaw_ref])
    eulers = np.array(eulers)
    eulers_ref = np.array(eulers_ref)
    
    plt.subplot(311)
    plt.plot(timestamp, eulers_ref[:, 0], color="black", linestyle="dashed", label="reference")
    plt.plot(timestamp, eulers[:, 0], color="blue", alpha=0.7)
    plt.ylabel("roll (deg)")
    plt.legend(loc="upper left")
    plt.xlim([min(timestamp), max(timestamp)])
    plt.ylim([-10, 10])
    plt.grid()

    plt.subplot(312)
    plt.plot(timestamp, eulers_ref[:, 1], color="black", linestyle="dashed", label="reference")
    plt.plot(timestamp, eulers[:, 1], color="blue", alpha=0.7)
    plt.ylabel("pitch (deg)")
    plt.xlim([min(timestamp), max(timestamp)])
    plt.ylim([-10, 10])
    plt.grid()

    plt.subplot(313)
    plt.plot(timestamp, eulers_ref[:, 2], color="black", linestyle="dashed", label="reference")
    plt.plot(timestamp, eulers[:, 2], color="blue", alpha=0.7)
    plt.ylabel("yaw (deg)")
    plt.xlim([min(timestamp), max(timestamp)])
    plt.grid()

    plt.xlabel("t (s)")

    plt.tight_layout()

    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')

def plot_xy_ate(data, ref_data, timestamp, savename=None):
    plt.cla()
    plt.clf()
    plt.close()
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

    ate = np.linalg.norm(data[:,:2,3] - ref_data[:,:2,3], ord=1, axis=1)
    #rmse = np.linalg.norm(data[:, :2] - ref_data[:, :2], ord=2, axis=1)
    #rmse = np.sqrt(np.mean((data[:, :2] - ref_data[:, :2])**2))

    plt.plot(timestamp, ate, color="gray", label="ATE (m)", zorder=1)

    plt.hlines(y=np.mean(ate), xmin=timestamp[0], xmax=timestamp[-1], label="mean", colors="red", zorder=2)
    plt.hlines(y=np.median(ate), xmin=timestamp[0], xmax=timestamp[-1], label="median", colors="green", zorder=2)
    plt.hlines(y=np.sqrt(np.mean(ate**2)), xmin=timestamp[0], xmax=timestamp[-1], label="rmse", colors="blue", zorder=2)

    plt.grid()
    plt.title("ATE")

    plt.xlim([min(timestamp), max(timestamp)])
    plt.ylim([0, max(2, max(ate))])
    plt.xlabel("t (s)")
    plt.ylabel("ATE (m)")
    plt.legend(loc="upper right")

    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')

def plot_yaw_error(data, ref_data, timestamp, savename=None):
    plt.cla()
    plt.clf()
    plt.close()
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

    eulers, eulers_ref = [], []
    for d, r in zip(data, ref_data):
        [yaw, pitch, roll] = R.from_matrix(d[:3,:3]).as_euler("zyx", degrees=True)
        [yaw_ref, pitch_ref, roll_ref] = R.from_matrix(r[:3,:3]).as_euler("zyx", degrees=True)
        eulers.append([roll, pitch, yaw])
        eulers_ref.append([roll_ref, pitch_ref, yaw_ref])
    eulers = np.array(eulers)
    eulers_ref = np.array(eulers_ref)

    ae = np.abs(eulers[:,2] - eulers_ref[:,2])

    plt.plot(timestamp, ae, color="gray", label="ATE (m)", zorder=1)

    plt.hlines(y=np.mean(ae), xmin=timestamp[0], xmax=timestamp[-1], label="mean", colors="red", zorder=2)
    plt.hlines(y=np.median(ae), xmin=timestamp[0], xmax=timestamp[-1], label="median", colors="green", zorder=2)
    plt.hlines(y=np.sqrt(np.mean(ae**2)), xmin=timestamp[0], xmax=timestamp[-1], label="rmse", colors="blue", zorder=2)

    plt.grid()
    plt.title("MAE")

    plt.xlim([min(timestamp), max(timestamp)])
    plt.ylim([0, max(5, max(ae))])
    plt.xlabel("t (s)")
    plt.ylabel("MAE (deg)")
    plt.legend(loc="upper right")

    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')

def plot_stats_summary():
    plt.cla()
    plt.clf()
    plt.close()
    pass


def plot_all(filename):
    plot_filename = filename.split(".")[0]
    plot_folder = f"plots/{plot_filename}"

    #shutil.rmtree(plot_folder)
    os.mkdir(plot_folder)

    filepath = "results/" + filename
    with open(filepath, "rb") as f:
        est = np.load(f)
        rtk = np.load(f)
        ts = np.load(f)

        # timestamp fo filename
        plot_xy_data(est, rtk, savename=f"{plot_folder}/xy_data.png")
        plot_xyz(est, rtk, ts, savename=f"{plot_folder}/xyz.png")
        plot_rpy(est, rtk, ts, savename=f"{plot_folder}/rpy.png")
        plot_xy_data_w_error(est, rtk, savename=f"{plot_folder}/xy_w_error.png")
        plot_xy_ate(est, rtk, ts, savename=f"{plot_folder}/xy_ate.png")
        plot_yaw_error(est, rtk, ts, savename=f"{plot_folder}/yaw_mae.png")


if __name__ == "__main__":
    pass
