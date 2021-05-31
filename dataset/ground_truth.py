import os
import scipy.io
import numpy as np


from ferryslam.dataset.sensor_dataset import DatasetType
from ferryslam.parameters import Parameters as par


# NOTE: this file must stay in the current folder get the folder location of this file!
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class GroundTruth:
    def __init__(self, x, y, z, scale=1):
        self.x = x
        self.y = y
        self.z = z
        self.scale = scale


def ground_truth_factory(dataset_type: DatasetType):
    if dataset_type == DatasetType.VIDEO:
        path = os.path.join(__location__, "videos/kitti06/groundtruth.txt")
        return VideoGroundTruth(path)
    if dataset_type == DatasetType.ROS:
        path = os.path.join(__location__, "groundtruths", par.gt_file_name)
        ned_origin_path = os.path.join(
            __location__, "groundtruths", par.ned_origin_file_name
        )
        return ROSGroundTruth(path, ned_origin_path)


class ROSGroundTruth:
    def __init__(self, path, ned_origin_path):
        self.path = path
        self.ned_origin_path = ned_origin_path

        # Read data file
        self.read_file()

        self.init = False

    def get_xyz(self, frame_id=None):
        """Get ground truth data"""
        # Get next element
        ss = next(self.tvecs_generator)
        x, y, z = float(ss[0]), float(ss[1]), float(ss[2])

        timestamp = next(self.timestamp_generator)
        """
        if not self.init:
            self.x_prev, self.y_prev, self.z_prev = x, y, z
            self.init = True
            return GroundTruth(0, 0, 0, 1)
        else:
            # Calculate scale
            scale = np.sqrt(
                (x - self.x_prev) ** 2 + (y - self.y_prev) ** 2 + (z - self.z_prev) ** 2
            )
            self.x_prev, self.y_prev, self.z_prev = x, y, z
            return GroundTruth(x, y, z, scale)
        """
        print("Obs scale is hardcoded to 1")
        return GroundTruth(x, y, z, 1)#, timestamp

    def read_file(self):
        # Read ground truth origin data
        ned_origin = scipy.io.loadmat(self.ned_origin_path)
        self.height0 = ned_origin["height0"].item()
        self.lat0 = ned_origin["lat0"].item()
        self.lon0 = ned_origin["lon0"].item()

        # Read ground truth pose data
        ground_truth = scipy.io.loadmat(self.path)

        timestamps = ground_truth["obsv_estimates"][0][0][0]
        navigaton_frame = ground_truth["obsv_estimates"][0][0][1]
        roll_hat = ground_truth["obsv_estimates"][0][0][2]
        pitch_hat = ground_truth["obsv_estimates"][0][0][3]
        yaw_hat = ground_truth["obsv_estimates"][0][0][4]

        omega_ib_b_hat = ground_truth["obsv_estimates"][0][0][5]
        ars_bias_hat = ground_truth["obsv_estimates"][0][0][6]
        ars_bias_total_hat = ground_truth["obsv_estimates"][0][0][7]
        acc_bias_hat = ground_truth["obsv_estimates"][0][0][8]
        gravity_hat = ground_truth["obsv_estimates"][0][0][9]
        tmo_innovation = ground_truth["obsv_estimates"][0][0][10]
        T_tmo_innovation = ground_truth["obsv_estimates"][0][0][11]
        T_tmo_innovation_sum = ground_truth["obsv_estimates"][0][0][12]

        p_Lb_L_hat = ground_truth["obsv_estimates"][0][0][13]

        v_eb_n_hat = ground_truth["obsv_estimates"][0][0][14]
        speed_course_hat = ground_truth["obsv_estimates"][0][0][15]
        innov = ground_truth["obsv_estimates"][0][0][16]
        innov_covariance = ground_truth["obsv_estimates"][0][0][17]
        P_hat = ground_truth["obsv_estimates"][0][0][18]

        # Compensate for time delay offset between rtk data and ros data
        self.timestamps = np.array(timestamps.T).astype("float") - par.rtk_bag_delay_s
        self.freq = 1 / np.diff(self.timestamps[:, 0]).mean()
        self.tvecs = p_Lb_L_hat.copy().T.astype("float")

        self.eulers = np.zeros((len(self.timestamps), 3)).astype("float")
        self.eulers[:, 0] = roll_hat.copy()
        self.eulers[:, 1] = pitch_hat.copy()
        self.eulers[:, 2] = yaw_hat.copy()

        # Create generators
        self.timestamp_generator = (t for t in self.timestamps)
        self.tvecs_generator = (tv for tv in self.tvecs)

        # return t, tvecs, eulers, v_eb_n_hat.T


class VideoGroundTruth:
    def __init__(self, path):
        self.path = path
        self.freq = 1

        with open(self.path) as f:
            self.data = f.readlines()

        self.init = False

    def get_xyz(self, frame_id):
        """Get ground truth data, based on image frame id"""
        if not self.init:
            self.init = True
            return GroundTruth(0, 0, 0, 1)
        if frame_id > 0:
            # Get data line
            ss = self.data[frame_id - 1].strip().split()
            x_prev, y_prev, z_prev = float(ss[0]), float(ss[1]), float(ss[2])
            # Get data line
            ss = self.data[frame_id].strip().split()
            x, y, z = float(ss[0]), float(ss[1]), float(ss[2])
            # Calculate scale
            scale = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)
            return GroundTruth(x, y, z, scale)
