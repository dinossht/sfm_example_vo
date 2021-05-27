import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *
from frontend import *
import gtsam


class Frame:
    def __init__(self, kp, des):
        self.kp = kp
        self.des = des
        self.matches = matches  # matches[(kf_idx, kp_idx)]


img1 = cv.imread("my_data/img1.png", cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread("my_data/img10.png", cv.IMREAD_GRAYSCALE)  # trainImage

feature = Feature()

kp1, des1 = feature.detectAndCompute(img1)
kp2, des2 = feature.detectAndCompute(img2)
print(f"num feat {len(kp1)} {len(kp2)}")

idx1, idx2, good = feature.goodMatches(des1, des2)
kp1_good = kp1[idx1]
kp2_good = kp2[idx2]
print(f"good match {len(kp1_good)} {len(kp2_good)}")

pt1_good = convertToPoints(kp1_good)
pt2_good = convertToPoints(kp2_good)

pt1_good_n = undistort_normalize(pt1_good)
pt2_good_n = undistort_normalize(pt2_good)

T1 = np.eye(4)  # Tw, this is world coordinate
T2, inliers = recoverPose(pt1_good_n, pt2_good_n)  
Twc = np.linalg.inv(T2)  # Trc/Twc is w.r.t ref/world coordinate frame
inlier_matches = good[inliers.ravel()==1]
pt1_inliers_n = pt1_good_n[inliers.ravel()==1]
pt2_inliers_n = pt2_good_n[inliers.ravel()==1]
print(f"inlier {len(inlier_matches)}")

pts3dw, good_pts_mask = triangulate_normalized_points(T2, T1, pt2_inliers_n, pt1_inliers_n)
print(f"points front {np.sum(pts3dw[:,2]>0)}")

# TODO: add this to filter bad points as well
pts3dw = pts3dw[pts3dw[:,2]>0]

print(f"\npose cur\n {Twc}")

img3 = cv.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

"""
plt.figure(), plt.scatter(pts3dw[:,0], pts3dw[:,2]), plt.xlim([-40, 40]), plt.ylim([0, 60])
plt.figure(), plt.imshow(img3)
plt.show()
"""

## BA 
# Variable symbols for camera poses.
X = gtsam.symbol_shorthand.X

# Variable symbols for observed 3D point "landmarks".
L = gtsam.symbol_shorthand.L

# Create a factor graph.
graph = gtsam.NonlinearFactorGraph()

# We will in this function assume that all keyframes have a common, given (constant) camera model.
calibration = gtsam.Cal3_S2(K[0,0], K[1,1], 0, K[0,2], K[1,2])

# Unfortunately, the dataset does not contain any information on uncertainty in the observations,
# so we will assume a common observation uncertainty of 3 pixels in u and v directions.
obs_uncertainty = gtsam.noiseModel.Isotropic.Sigma(2, 3.0)

# Extract the first two keyframes (which we will assume has ids 0 and 1).
kf_0 = sfm_map.get_keyframe(0)
kf_1 = sfm_map.get_keyframe(1)

"""
# Add measurements.
for keyframe in sfm_map.get_keyframes():
    for keypoint_id, map_point in keyframe.get_observations().items():
        obs_point = keyframe.get_keypoint(keypoint_id).point()
        factor = gtsam.GenericProjectionFactorCal3_S2(obs_point, obs_uncertainty,
                                                        X(keyframe.id()), L(map_point.id()),
                                                        calibration)
        graph.push_back(factor)

# Set prior on the first camera (which we will assume defines the reference frame).
no_uncertainty_in_pose = gtsam.noiseModel.Constrained.All(6)
factor = gtsam.PriorFactorPose3(X(kf_0.id()), gtsam.Pose3(), no_uncertainty_in_pose)
graph.push_back(factor)

# Set prior on distance to next camera.
no_uncertainty_in_distance = gtsam.noiseModel.Constrained.All(1)
prior_distance = 1.0
factor = gtsam.RangeFactorPose3(X(kf_0.id()), X(kf_1.id()), prior_distance, no_uncertainty_in_distance)
graph.push_back(factor)

# Set initial estimates from map.
initial_estimate = gtsam.Values()
for keyframe in sfm_map.get_keyframes():
    pose_w_c = gtsam.Pose3(keyframe.pose_w_c().to_matrix())
    initial_estimate.insert(X(keyframe.id()), pose_w_c)

for map_point in sfm_map.get_map_points():
    point_w = gtsam.Point3(map_point.point_w().squeeze())
    initial_estimate.insert(L(map_point.id()), point_w)

# Optimize the graph.
params = gtsam.GaussNewtonParams()
params.setVerbosity('TERMINATION')
optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, params)
print('Optimizing:')
result = optimizer.optimize()
"""