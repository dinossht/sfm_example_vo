import numpy as np
import cv2 as cv


K = np.array([
    [1.9954e+03, 0.0000e+00, 9.6550e+02],
    [0.0000e+00, 1.9952e+03, 6.0560e+02],
    [0.0000e+00, 0.0000e+00, 1.0000e+00]])
Kinv = np.array([
    [ 5.01152651e-04,  0.00000000e+00, -4.83862885e-01],
    [ 0.00000000e+00,  5.01202887e-04, -3.03528468e-01],
    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
D = np.array([-0.14964, 0.13337, 0., 0., 0.])

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def undistort(pts):
    undistorted = cv.undistortPoints(pts, K, D, None, K)
    return undistorted.ravel().reshape(undistorted.shape[0], 2)

def normalize(pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def undistort_normalize(pts):
    pts = undistort(pts)
    pts = normalize(pts)
    return pts

def triangulate_normalized_points(pose_1w, pose_2w, kpn_1, kpn_2):
    # P1w = np.dot(K1,  M1w) # K1*[R1w, t1w]
    # P2w = np.dot(K2,  M2w) # K2*[R2w, t2w]
    # since we are working with normalized coordinates x_hat = Kinv*x, one has         
    P1w = pose_1w[:3,:] # [R1w, t1w]
    P2w = pose_2w[:3,:] # [R2w, t2w]

    point_4d_hom = cv.triangulatePoints(P1w, P2w, kpn_1.T, kpn_2.T)
    good_pts_mask = np.where(point_4d_hom[3]!= 0)[0]
    point_4d = point_4d_hom / point_4d_hom[3] 
    
    points_3d = point_4d[:3, :]
    return points_3d, good_pts_mask

def convertToPoints(kps):
    pts = []
    for k in kps:
        pts.append(k.pt)
    return np.array(pts)
