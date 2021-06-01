import cv2
import numpy as np
from pylie import SO3, SE3
from sfm_map import MatchedFrame


def triangulate_points_from_two_views(P_0, kp_0: np.ndarray,
                                      P_1, kp_1: np.ndarray):
    # Triangulate wrt frame 0.
    points_hom = cv2.triangulatePoints(P_0, P_1, kp_0, kp_1)
    assert np.sum(points_hom[-1, :] != 0) != 0, "division by zero somewhere"
    return points_hom[:-1, :] / points_hom[-1, :]

def estimate_pose_from_map_correspondences(K, kp: np.ndarray, points_w: np.ndarray):
    # Estimate initial pose with a (new) PnP-method.
    _, theta_vec, t = cv2.solvePnP(points_w.T, kp.T, K, None)#, flags=cv2.SOLVEPNP_SQPNP)
    pose_c_w = SE3((SO3.Exp(theta_vec), t.reshape(3, 1)))

    return pose_c_w.inverse()