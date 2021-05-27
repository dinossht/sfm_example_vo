import cv2
import numpy as np
from pylie import SO3, SE3
from dataset import read_dataset
from sfm_map import SfmMap, Keyframe, MapPoint, MatchedFrame, PerspectiveCamera, FeatureTrack, KeyPoint
import open3d as o3d
from optimize import BatchBundleAdjustment, IncrementalBundleAdjustment
from utils import *
from frontend import *
import matplotlib.pyplot as plt

# TODO: remove bad map points after optimization

f = np.array([[1.9954e+03, 1.9952e+03]]).T
principal_point = np.array([[9.6550e+02, 6.0560e+02]]).T

"""Assumes undistorted pixel points"""

def estimate_two_view_relative_pose(frame0: MatchedFrame, kp_0: np.ndarray,
                                    frame1: MatchedFrame, kp_1: np.ndarray):
    num_matches = kp_0.shape[1]
    if num_matches < 8:
        return None

    # Compute fundamental matrix from matches.
    F_0_1, _ = cv2.findFundamentalMat(kp_1.T, kp_0.T, cv2.FM_8POINT)

    # Extract the calibration matrices.
    K_0 = frame0.camera_model().calibration_matrix()
    K_1 = frame1.camera_model().calibration_matrix()

    # Compute the essential matrix from the fundamental matrix.
    E_0_1 = K_0.T @ F_0_1 @ K_1

    # Compute the relative pose.
    # Transform detections to normalized image plane (since cv2.recoverPose() only supports common K)
    kp_n_0 = frame0.camera_model().pixel_to_normalised(kp_0)
    kp_n_1 = frame1.camera_model().pixel_to_normalised(kp_1)
    K_n = np.identity(3)
    _, R_0_1, t_0_1, _ = cv2.recoverPose(E_0_1, kp_n_1.T, kp_n_0.T, K_n)

    return SE3((SO3(R_0_1), t_0_1))

def triangulate_points_from_two_views(P_0: MatchedFrame, kp_0: np.ndarray,
                                      P_1: MatchedFrame, kp_1: np.ndarray):
    # Triangulate wrt frame 0.
    points_hom = cv2.triangulatePoints(P_0, P_1, kp_0, kp_1)
    return points_hom[:-1, :] / points_hom[-1, :]

def estimate_pose_from_map_correspondences(K, kp: np.ndarray, points_w: np.ndarray):
    # Estimate initial pose with a (new) PnP-method.
    _, theta_vec, t = cv2.solvePnP(points_w.T, kp.T, K, None, flags=cv2.SOLVEPNP_SQPNP)
    pose_c_w = SE3((SO3.Exp(theta_vec), t.reshape(3, 1)))

    return pose_c_w.inverse()

def interactive_isfm():
    # Choose optimization method, BatchBundleAdjustment 
    optimizer = BatchBundleAdjustment()

    img_paths = [   "/home/dino/Code/masterproject/sfm_example_vo/my_data/img1.png",
                    "/home/dino/Code/masterproject/sfm_example_vo/my_data/img10.png",
                    "/home/dino/Code/masterproject/sfm_example_vo/my_data/img20.png"]
    
    frame_0 = MatchedFrame(0, PerspectiveCamera(f, principal_point), img_paths[0])
    frame_1 = MatchedFrame(1, PerspectiveCamera(f, principal_point), img_paths[1])
    frame_2 = MatchedFrame(2, PerspectiveCamera(f, principal_point), img_paths[2])

    sfm_map = initialize_map(frame_0, frame_1, frame_2)

    def get_geometry():
        poses = sfm_map.get_keyframe_poses()
        p, c = sfm_map.get_pointcloud()

        axes = []
        for pose in poses:
            axes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0).transform(pose.to_matrix()))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p.T)
        pcd.colors = o3d.utility.Vector3dVector(c.T / 255)

        return [pcd] + axes

    def optimize(vis):
        points3d = np.array(list(sfm_map.get_map_points()))
        points3d = np.array([p._point_w for p in points3d]).squeeze()
        plt.scatter(points3d[:,0], points3d[:,2])

        optimizer.full_bundle_adjustment_update(sfm_map)
        
        points3d = np.array(list(sfm_map.get_map_points()))
        points3d = np.array([p._point_w for p in points3d]).squeeze()
        plt.scatter(points3d[:,0], points3d[:,2])
        plt.show()

        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)

    # Create visualizer.
    key_to_callback = {}
    key_to_callback[ord("O")] = optimize
    o3d.visualization.draw_geometries_with_key_callbacks(get_geometry(), key_to_callback)

def initialize_map(frame_0, frame_1, frame_2):
    img0 = cv.imread("my_data/img1.png", cv.IMREAD_GRAYSCALE)  # queryImage
    img1 = cv.imread("my_data/img10.png", cv.IMREAD_GRAYSCALE)  # trainImage

    feature = Feature(2000)

    kp0, des0 = feature.detectAndCompute(img0)
    kp1, des1 = feature.detectAndCompute(img1)
    print(f"num feat {len(kp0)} {len(kp1)}")

    idx0, idx1, good = feature.goodMatches(des0, des1)
    print(f"good match {len(good)}")

    pt0_good_n = undistort_normalize(convertToPoints(kp0[idx0]))
    pt1_good_n = undistort_normalize(convertToPoints(kp1[idx1]))

    T_0_1, inliers = recoverPose(pt1_good_n, pt0_good_n)  
    pose_0_1 = SE3((SO3(T_0_1[:3,:3]), T_0_1[:3,3])) # pose_w_c, 0 == world_frame, 1 == cam_frame

    Kp_0 = convertToPoints(kp0[idx0]).T
    Kp_1 = convertToPoints(kp1[idx1]).T

    inlier_matches = good[inliers.ravel()==1]
    id_0 = idx0[inliers.ravel()==1]
    id_1 = idx1[inliers.ravel()==1]
    kp_0 = convertToPoints(kp0)[id_0].T
    kp_1 = convertToPoints(kp1)[id_1].T
    print(f"inlier {len(inlier_matches)}")

    P_0 = frame_0.camera_model().projection_matrix(SE3())
    P_1 = frame_1.camera_model().projection_matrix(pose_0_1.inverse())
    points_0 = triangulate_points_from_two_views(P_0, kp_0, P_1, kp_1)

    # Filter depth
    #depth_mask = points_0[2,:]>0
    depth_mask = np.logical_and(points_0[2,:]>0, points_0[2,:] < 100)
    points_0 = points_0[:,depth_mask]
    id_0 = id_0[depth_mask]
    id_1 = id_1[depth_mask]

    # Third frame
    img2 = cv.imread("my_data/img20.png", cv.IMREAD_GRAYSCALE) 
    kp2, des2 = feature.detectAndCompute(img2)

    # input: 3d descriptor, new 2d descriptor
    des0_3d = des0[id_0]
    des1_3d = des0[id_1]
    kp0_3d = kp0[id_0]
    kp1_3d = kp0[id_1]
    idx_3d, idx_2, good = feature.goodMatches(des0_3d, des2)

    # output: matches
    id_0 = id_0[idx_3d]
    id_1 = id_1[idx_3d]
    id_2 = idx_2
    points_0_new = points_0[:, idx_3d]

    kp_new = convertToPoints(kp2[id_2]).T
    pose_0_2 = estimate_pose_from_map_correspondences(K, kp_new, points_0_new) # pose_w_c, w == world_frame, new == cam_frame

    sfm_map = SfmMap()

    # Add first keyframe as reference frame.
    kf_0 = Keyframe(frame_0, SE3())
    sfm_map.add_keyframe(kf_0)
    # Add second keyframe from relative pose.
    kf_1 = Keyframe(frame_1, pose_0_1)
    sfm_map.add_keyframe(kf_1)
    # Add third keyframe from relative pose.
    kf_2 = Keyframe(frame_2, pose_0_2)
    sfm_map.add_keyframe(kf_2)

    matched_frames = [frame_0, frame_1, frame_2]
    ids = np.array([id_0, id_1, id_2])
    kps = np.array([kp0, kp1, kp2], dtype=object)
    num_points = len(id_0)  # num matches

    img = matched_frames[0].load_image()
    for i in range(num_points):
        
        curr_track = FeatureTrack()
        curr_map_point = MapPoint(i, points_0_new[:, [i]])

        for c in range(3):
            cam_ind = c
            det_id = ids[c, i]
            
            det_point = np.array([kps[c][det_id].pt]).T
            color = np.reshape(img[int(det_point[1]),int(det_point[0])], (3,1))

            curr_track.add_observation(matched_frames[cam_ind], det_id)
            matched_frames[cam_ind].add_keypoint(det_id, KeyPoint(det_point, color, curr_track))
            curr_map_point.add_observation(sfm_map.get_keyframe(cam_ind), det_id)

        sfm_map.add_map_point(curr_map_point)

    for frame in matched_frames:
        frame.update_covisible_frames()


    """
    # TODO: how to handle new map point ids? Do new point triangulation?
    # Third frame
    img3 = cv.imread("my_data/img30.png", cv.IMREAD_GRAYSCALE) 
    kp3, des3 = feature.detectAndCompute(img3)

    # input: 3d descriptor, new 2d descriptor
    idx_3d, idx_3, good = feature.goodMatches(des0_3d, des3)

    # output: matches
    id_3 = idx_3
    points_0_new = points_0[:, idx_3d]

    kp_new = convertToPoints(kp3[id_3]).T
    pose_0_3 = estimate_pose_from_map_correspondences(K, kp_new, points_0_new) # pose_w_c, w == world_frame, new == cam_frame

    # Add third keyframe from relative pose.
    kf_3 = Keyframe(frame_3, pose_0_3)
    sfm_map.add_keyframe(kf_3)
    """

    return sfm_map
    

interactive_isfm()