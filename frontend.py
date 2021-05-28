import numpy as np
import cv2 as cv
from utils import *
import matplotlib.pyplot as plt
from orbslam2_features import ORBextractor


class Feature:
    def __init__(self, num_features):
        self.orb2 = ORBextractor(num_features, 1.2, 8)  #num feat, scale, levels
        self.bf = cv.BFMatcher()

    def detectAndCompute(self, img):
        kps_tuples, des = self.orb2.detectAndCompute(img)
        WIDTH_MASK, HEIGHT_MASK = 590, 850
        kps_tuples_filtered, des_filtered = [], []
        for i in range(len(kps_tuples)):
            if not (kps_tuples[i][0] < WIDTH_MASK and kps_tuples[i][1] > HEIGHT_MASK):  # metal rail mask
                kps_tuples_filtered.append(kps_tuples[i])
                des_filtered.append(des[i])
        kps_tuples = kps_tuples_filtered
        des = np.array(des_filtered)
        
        # convert to keypoints 
        kps = [cv.KeyPoint(*kp) for kp in kps_tuples]
        return np.array(kps), np.array(des)

    def goodMatches(self, des1, des2):
        matches = self.bf.knnMatch(des1, des2, k=2)
        idx1, idx2 = [], []
        good = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                good.append(m)
        return np.array(idx1), np.array(idx2), np.array(good)

def recoverPose(pts_ref, pts_cur):
    E, inliers = cv.findEssentialMat(pts_ref, pts_cur, focal=1, pp=(0.,0.), method=cv.RANSAC, prob=0.999, threshold=0.0003) 
    _, R, t, _ = cv.recoverPose(E, pts_ref, pts_cur, focal=1, pp=(0., 0.))
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t.ravel()

    return T, inliers