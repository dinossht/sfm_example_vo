import os
import scipy.io
import numpy as np


class GroundTruth:
    def __init__(self, x, y, z, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


class ROSGroundTruth:
    def __init__(self, path, ned_origin_path, trip_nr):
        self.trip_nr = trip_nr
        self.path = path
        self.ned_origin_path = ned_origin_path

        # Read data file
        self.read_file()

    def get_xyz(self, timestamp):
        """Get ground truth data"""
        i = (np.abs(self.timestamps - timestamp)).argmin() 

        x, y, z = self.tvecs[i]
        roll, pitch, yaw = self.eulers[i]
        return GroundTruth(x, y, z, roll, pitch, yaw)

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
        rtk_bag_delay_s = [18.24, 18.33, 18.42, 18.54][self.trip_nr - 1]
        self.timestamps = np.array(timestamps.T).astype("float") - rtk_bag_delay_s
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