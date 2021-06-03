import numpy as np
import cv2 as cv
from utils import *
import matplotlib.pyplot as plt
from orbslam2_features import ORBextractor
from collections import defaultdict


class Feature:
    def __init__(self, num_features, lowes_ratio=0.7):
        self.orb2 = ORBextractor(num_features, 1.2, 16)  #num feat, scale, levels
        self.bf = cv.BFMatcher()
        self.lowes_ratio = lowes_ratio

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

    def match(self, des1, des2):
        idx1, idx2, good =  self.goodMatchesOneToOne(des1, des2)
        return np.array(idx1), np.array(idx2), np.array(good)

    def goodMatches(self, des1, des2):
        matches = self.bf.knnMatch(des1, des2, k=2)
        idx1, idx2 = [], []
        good = []
        for m,n in matches:
            if m.distance < self.lowes_ratio * n.distance:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                good.append(m)
        return np.array(idx1), np.array(idx2), np.array(good)

    def goodMatchesOneToOne(self, des1, des2):
        matches = self.bf.knnMatch(des1, des2, k=2)
        len_des2 = len(des2)
        idx1, idx2 = [], []  
        if matches is not None:         
            float_inf = float('inf')
            dist_match = defaultdict(lambda: float_inf)   
            index_match = dict()  
            for m, n in matches:
                if m.distance > self.lowes_ratio * n.distance:
                    continue     
                dist = dist_match[m.trainIdx]
                if dist == float_inf: 
                    # trainIdx has not been matched yet
                    dist_match[m.trainIdx] = m.distance
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    index_match[m.trainIdx] = len(idx2)-1
                else:
                    if m.distance < dist: 
                        # we have already a match for trainIdx: if stored match is worse => replace it
                        #print("double match on trainIdx: ", m.trainIdx)
                        index = index_match[m.trainIdx]
                        assert(idx2[index] == m.trainIdx) 
                        idx1[index]=m.queryIdx
                        idx2[index]=m.trainIdx                        
        return np.array(idx1), np.array(idx2), None

def recoverPose(pts_ref, pts_cur):
    E, inliers = cv.findEssentialMat(pts_ref, pts_cur, focal=1, pp=(0.,0.), method=cv.RANSAC, prob=0.999, threshold=0.0003) 
    _, R, t, _ = cv.recoverPose(E, pts_ref, pts_cur, focal=1, pp=(0., 0.))
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t.ravel()

    return T, inliers