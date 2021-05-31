import numpy as np
import cv2
from frontend import *   
from pylie import SO3, SE3
   
i = 0
N = 20
off = 0
def next_frame_path():
    global i, N, off
    path = f"kaia_data/frame_id_{off + i * N}.png"
    frame_id = i
    i += 1
    return path, frame_id

class MyFeatTrack:
    def __init__(self, kps, des):
        self.des = des
        self.kps_raw = kps
        self.kps = convertToPoints(kps)
        self.kps_n = undistort_normalize(self.kps)

        self.num_kps = len(kps)

        self.idxs = np.array(list(range(len(kps))))
        self.good_idxs = self.idxs.copy()

        self.pts3d_w = np.array([None] * self.num_kps)

    def good_des(self):
        return self.des[self.good_idxs]

    def good_kps(self):
        return self.kps[self.good_idxs]

    def good_kps_n(self):
        return self.kps_n[self.good_idxs]

class calibrate:
    def __init__(self):
        self.feature = Feature(5000, 0.7)
        self.NUM_ITER = 50
   
    def initialize(self):
        # Load images
        path0, _ = next_frame_path()
        path1, _ = next_frame_path()
        img0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

        headings = []
        yaws = []
        rolls = []
        for it in range(self.NUM_ITER):
            # Detect features
            kp0, des0 = self.feature.detectAndCompute(img0)
            kp1, des1 = self.feature.detectAndCompute(img1)
            feat_track = [None] * 2
            feat_track[0] = MyFeatTrack(kp0, des0)
            feat_track[1] = MyFeatTrack(kp1, des1)
            print(f"Num feat: {len(feat_track[0].good_idxs)}")

            # Match features
            good_idx0, good_idx1, _ = self.feature.goodMatches(des0, des1)
            feat_track[0].good_idxs = feat_track[0].good_idxs[good_idx0]
            feat_track[1].good_idxs = feat_track[1].good_idxs[good_idx1]
            print(f"Num good match: {len(feat_track[0].good_idxs)}")

            # Estimate pose 
            T_0_1, inliers_mask = recoverPose(feat_track[1].good_kps_n(), feat_track[0].good_kps_n())  
            pose_0_1 = SE3((SO3(T_0_1[:3,:3]), T_0_1[:3,3])) # pose_w_c, 0 == world_frame, 1 == cam_frame

            # Calculate initial yaw angle offset
            heading_vector = pose_0_1.translation
            heading_vector[:2] = 0
            reference_vector = np.array([[0], [0], [1]])  # initially pointing forward with z
            heading = 180 * np.arccos(np.dot(heading_vector, reference_vector[:, 0])) / np.pi
            headings.append(heading)

            # Calculate initial yaw angle offset
            yaw_vector = np.zeros((3,))
            yaw_vector[1] = pose_0_1.translation[1]
            reference_vector = np.array([[0], [1], [0]])  # initially pointing forward with z
            yaw = 180 * np.arccos(np.dot(yaw_vector, reference_vector[:, 0])) / np.pi
            yaws.append(yaw)

            # Calculate initial pitch angle offset
            roll_vector = np.zeros((3,))
            roll_vector[1] = pose_0_1.translation[0]
            reference_vector = np.array([[1], [0], [0]])  # initially pointing forward with z
            roll = 180 * np.arccos(np.dot(roll_vector, reference_vector[:, 0])) / np.pi
            rolls.append(roll)
            print(it)

        plt.subplot(311)
        plt.hist(headings, bins=100, range=(0, 20))
        plt.subplot(312)
        plt.hist(yaws, bins=100, range=(90-3, 90+3))
        plt.subplot(313)
        plt.hist(rolls, bins=100, range=(90-3, 90+3))
        print(np.median(headings))
        print(np.median(yaws))
        print(np.median(rolls))
        plt.show()
    
c = calibrate()
c.initialize()