from sfm_map import SfmMap, Keyframe, MapPoint, KeyPoint, MatchedFrame, PerspectiveCamera, FeatureTrack
from pylie import SO3, SE3
from frontend import *
from utils import *
from sfm_frontend_utils import *
from parameters import param
import gtsam


MAX_DEPTH = 200 # in meter
MIN_DEPTH = 0.5
MIN_NUM_OBS = 3
MAX_PIXEL_ERR = 5
FILT_DEPTH = True
cos_max_parallax = 0.99999

class UniquePoint3D:
    global_point_id = 0
    def __init__(self, point):
        self.point = np.reshape(point, (3,1))

        # Add unique id to 3d point
        self.id = UniquePoint3D.global_point_id
        UniquePoint3D.global_point_id += 1


class MyFeatTrack:
    def __init__(self, kps, des):
        self.des = des
        self.kps_raw = kps
        self.kps = undistort(convertToPoints(kps))
        self.kps_n = normalize(self.kps)

        self.num_kps = len(kps)

        self.idxs = np.array(list(range(len(kps))))
        self.good_idxs = self.idxs.copy()

        self.pts3d_w = np.array([None] * self.num_kps)

    def good_des(self):
        return self.des[self.good_idxs]

    def good_kps_raw(self):
        return self.kps_raw[self.good_idxs]

    def good_kps(self):
        return self.kps[self.good_idxs]

    def good_kps_n(self):
        return self.kps_n[self.good_idxs]

    def good_pts3d_w(self):
        return self.pts3d_w[self.good_idxs]

    def set_points3d(self, point3d: UniquePoint3D):
        self.pts3d_w[self.good_idxs] = point3d
    
    def get_plot_points3d(self):
        return np.array([p.point for p in self.pts3d_w[self.good_idxs]])


class SFM_frontend:
    def __init__(self, num_feat=2000, lowes_ratio=0.7):
        self.feature = Feature(num_feat, lowes_ratio)

        # Hardcoded
        self.f = np.array([[1.9954e+03, 1.9952e+03]]).T
        self.principal_point = np.array([[9.6550e+02, 6.0560e+02]]).T
        self.K = K

    def initialize(self, img0_path=None, img1_path=None, img0=None, img1=None, rtk_pose0=None, rtk_pose1=None, ts0=None, ts1=None, IMU_data=None, IMU_times=None, GNSS_data=None, GNSS_times=None):
        print("Initializing")
        print("#" * 20)
        # Init match frames
        matched_frames = [
            MatchedFrame(0, PerspectiveCamera(self.f, self.principal_point), img0_path, img0),
            MatchedFrame(1, PerspectiveCamera(self.f, self.principal_point), img1_path, img1)]

        # Load images
        img0 = matched_frames[0].load_image_grayscale()
        img1 = matched_frames[1].load_image_grayscale()

        # Detect features
        kp0, des0 = self.feature.detectAndCompute(img0)
        kp1, des1 = self.feature.detectAndCompute(img1)
        feat_track = [None] * 2
        feat_track[0] = MyFeatTrack(kp0, des0)
        feat_track[1] = MyFeatTrack(kp1, des1)
        print(f"Num feat: {len(feat_track[0].good_idxs)}")

        # Match features
        good_idx0, good_idx1, _ = self.feature.match(des0, des1)
        feat_track[0].good_idxs = feat_track[0].good_idxs[good_idx0]
        feat_track[1].good_idxs = feat_track[1].good_idxs[good_idx1]
        print(f"Num good match: {len(feat_track[0].good_idxs)}")

        # Estimate pose 
        T_0_1, inliers_mask = recoverPose(feat_track[1].good_kps_n(), feat_track[0].good_kps_n())  
        pose_0_1 = SE3((SO3(T_0_1[:3,:3]), T_0_1[:3,3])) # pose_w_c, 0 == world_frame, 1 == cam_frame

        # Provide camera pose in ned frame
        tB_b_c = np.array([3.285, 0, -1.4])  # camera origo given in body frame
        R_xyz_zxy = np.array([[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]])
        R_b_c = R_z(13) @ R_y(-1.75) @ R_xyz_zxy
        T_b_c = np.eye(4)
        T_b_c[:3,:3] = R_b_c
        T_b_c[:3,3] = tB_b_c  # pose of camera in body frame

        T_n_b = rtk_pose0     # pose of body in ned frame
        T_n_c = T_n_b @ T_b_c

        T0 = T_n_c @ SE3().to_matrix()
        T1 = T_n_c @ T_0_1
        pose0 = SE3((SO3(T0[:3,:3]), T0[:3,3]))
        pose1 = SE3((SO3(T1[:3,:3]), T1[:3,3]))

        # Mask inliers kps
        feat_track[0].good_idxs = feat_track[0].good_idxs[inliers_mask.ravel()==1]
        feat_track[1].good_idxs = feat_track[1].good_idxs[inliers_mask.ravel()==1]
        print(f"Num inliers: {len(feat_track[0].good_idxs)}")

        # Triangulate map points
        P_0 = matched_frames[0].camera_model().projection_matrix(pose0.inverse())
        P_1 = matched_frames[1].camera_model().projection_matrix(pose1.inverse())
        points_0 = triangulate_points_from_two_views(P_0, feat_track[0].good_kps().T, P_1, feat_track[1].good_kps().T)

        # Convert to ned
        #points_0 = (self.T_imu_c @ add_ones(points_0.T).T)[:3,:]

        # Create 3d points with unique id
        # Add 3d points to match feat0
        points_0_obj = [UniquePoint3D(p) for p in points_0.T].copy()
        feat_track[0].set_points3d(points_0_obj.copy())
        feat_track[1].set_points3d(points_0_obj.copy())

   

        if FILT_DEPTH:
            ## Filter depth, using kf0 and points given in its camera coordinate
            T_c_w = pose0.inverse().to_matrix()
            points_c = (T_c_w @ add_ones(points_0.T).T)[:3,:]

            # Remove points with negative depth for kf_0
            positive_depth_idxs = points_c[2,:] > MIN_DEPTH

            # Remove long distance points
            max_depth_mask = points_c[2,:] < MAX_DEPTH

            depth_mask = np.logical_and(positive_depth_idxs, max_depth_mask)
            feat_track[0].good_idxs = feat_track[0].good_idxs[depth_mask]
            feat_track[1].good_idxs = feat_track[1].good_idxs[depth_mask]
            print(f"Num good depth: {len(feat_track[0].good_idxs)}")
            ##
    
        # Reprojection error
        pt3d_h0 = add_ones(feat_track[0].get_plot_points3d().squeeze())
        uv_hat0 = P_0 @ pt3d_h0.T
        uv_hat0 = (uv_hat0[:2]/uv_hat0[-1]).T
        uv0 = feat_track[0].good_kps()

        pt3d_h1 = add_ones(feat_track[1].get_plot_points3d().squeeze())
        uv_hat1 = P_1 @ pt3d_h1.T
        uv_hat1 = (uv_hat1[:2]/uv_hat1[-1]).T
        uv1 = feat_track[1].good_kps()

        good0 = np.linalg.norm(uv0-uv_hat0, axis=1) < MAX_PIXEL_ERR
        good1 = np.linalg.norm(uv1-uv_hat1, axis=1) < MAX_PIXEL_ERR
        good_mask = np.logical_and(good0, good1)
        feat_track[0].good_idxs = feat_track[0].good_idxs[good_mask]
        feat_track[1].good_idxs = feat_track[1].good_idxs[good_mask]
        assert len(feat_track[0].good_idxs) == len(feat_track[1].good_idxs)  # unique indices check
        print(f"Num after pixel error: {len(feat_track[0].good_idxs)}")

        # Parallax filter
        # Compute back-projected rays (unit vectors) 
        rays1 = np.dot(pose0._rotation._matrix, add_ones(feat_track[0].good_kps_n()).T).T  # vector from keyframe origin to normalized point given in world frame
        norm_rays1 = np.linalg.norm(rays1, axis=-1, keepdims=True)                  
        rays1 /= norm_rays1                      

        rays2 = np.dot(pose1._rotation._matrix, add_ones(feat_track[1].good_kps_n()).T).T  # vector from keyframe origin to normalized point given in world frame
        norm_rays2 = np.linalg.norm(rays2, axis=-1, keepdims=True)  
        rays2 /= norm_rays2 

        # Compute dot products of rays. a dot b = |a|*|b|*cos(angle) = 1*1*cos(angle)
        cos_parallaxs = np.sum(rays1 * rays2, axis=1)  
        # Max parallax is almost 180 degrees, and min parallax angle is almost 0 degrees
        good_cos_parallaxs = np.logical_and(cos_parallaxs > 0, cos_parallaxs < cos_max_parallax)
        feat_track[0].good_idxs = feat_track[0].good_idxs[good_cos_parallaxs]
        feat_track[1].good_idxs = feat_track[1].good_idxs[good_cos_parallaxs]
        assert len(feat_track[0].good_idxs) == len(feat_track[1].good_idxs)  # unique indices check
        print(f"Num after parallax filt: {len(feat_track[0].good_idxs)}")

        # Filter non-unique idxs
        unique_idxs = return_unique_mask(feat_track[1].good_idxs)
        feat_track[0].good_idxs = feat_track[0].good_idxs[unique_idxs]
        feat_track[1].good_idxs = feat_track[1].good_idxs[unique_idxs]
        assert len(set(feat_track[0].good_idxs)) == len(set(feat_track[1].good_idxs))  # unique indices check
        print(f"Num unique: {len(feat_track[0].good_idxs)}")
        

        sfm_map = SfmMap()
        sfm_map.IMU_data = IMU_data
        sfm_map.IMU_times = IMU_times
        sfm_map.GNSS_data = GNSS_data
        sfm_map.GNSS_times = GNSS_times

        # Add first keyframe as reference frame.
        kf_0 = Keyframe(matched_frames[0], pose0, rtk_pose0, ts0)
        sfm_map.add_keyframe(kf_0)
        # Add second keyframe from relative pose.
        kf_1 = Keyframe(matched_frames[1], pose1, rtk_pose1, ts1)
        sfm_map.add_keyframe(kf_1)

        # Add keyframe1 intial bias and velocity
        current_vel = (kf_1.rtk_pose[:3,3] - kf_0.rtk_pose[:3,3]) / (kf_1.ts - kf_0.ts) 
        current_bias = gtsam.imuBias.ConstantBias(np.zeros((3,)), np.zeros((3,)))  # acc, gyro
        
        kf_0.current_imu_pose = gtsam.Pose3(kf_0.rtk_pose)
        kf_0.current_vel = current_vel
        kf_0.current_bias = current_bias
        kf_1.current_imu_pose = gtsam.Pose3(kf_1.rtk_pose)
        kf_1.current_vel = current_vel
        kf_1.current_bias = current_bias

        # Calculate initial yaw angle offset
        heading_vector = kf_1.pose_w_c().translation - kf_0.pose_w_c().translation
        heading_vector /= np.linalg.norm(heading_vector)
        heading_vector[:2] = 0
        reference_vector = np.array([[0], [0], [1]])  # initially pointing forward with z
        print(f"Init camera yaw mount offset: {180 * np.arccos(np.dot(heading_vector, reference_vector[:, 0])) / np.pi}")

        color_img = matched_frames[0].load_image()

        latest_map_points = []
        num_points = len(feat_track[0].good_idxs)
        for i in range(num_points):
            curr_track = FeatureTrack()
            curr_map_point_id = feat_track[0].good_pts3d_w()[i].id
            curr_map_point_coord = feat_track[0].good_pts3d_w()[i].point
            curr_map_point_des = feat_track[0].good_des()[i]  # first frame is the keyframe, so add it's descriptor
            curr_map_point_kps_raw = feat_track[0].good_kps_raw()[i]  # first frame is the keyframe, so add it's keypoint
            curr_map_point = MapPoint(curr_map_point_id, curr_map_point_coord, curr_map_point_des, curr_map_point_kps_raw)

            for cam_ind in range(len(matched_frames)):
                det_id = feat_track[cam_ind].good_idxs[i]
                det_point = feat_track[cam_ind].good_kps()[i].reshape(2,1)

                if int(det_point[1]) < color_img.shape[0] and int(det_point[1]) >= 0 and int(det_point[0]) >= 0 and int(det_point[0]) < color_img.shape[1]:
                    color = np.reshape(color_img[int(det_point[1]), int(det_point[0])], (3,1))
                else:
                    color = np.zeros((3,1))

                curr_track.add_observation(matched_frames[cam_ind], det_id)
                matched_frames[cam_ind].add_keypoint(det_id, KeyPoint(det_point, color, curr_track))
                curr_map_point.add_observation(sfm_map.get_keyframe(cam_ind), det_id)

            latest_map_points.append(curr_map_point)
            sfm_map.add_map_point(curr_map_point)

        sfm_map.set_latest_map_points(latest_map_points)
        return sfm_map

    def track_map(self, sfm_map, img_path=None, img=None, rtk_pose=None, ts=None):
        print("Tracking")
        print("#" * 20)
        frame_idx = sfm_map._cur_keyframe_id + 1
        matched_frame = MatchedFrame(frame_idx, PerspectiveCamera(self.f, self.principal_point), img_path, img)

        img = matched_frame.load_image_grayscale()
        kp, des = self.feature.detectAndCompute(img)

        # Extract map points
        des_map, kp_raw_map, points3d_map = [], [], []
        for map_point in sfm_map.get_latest_map_points():
            des_map.append(map_point._des)
            kp_raw_map.append(map_point._kps_raw)
            points3d_map.append(map_point.point_w())

        des_map = np.array(des_map)
        kp_raw_map = np.array(kp_raw_map)
        points3d_map = np.array(points3d_map).squeeze()

        feat_track = [None] * 2
        feat_track[0] = MyFeatTrack(kp_raw_map, des_map)
        feat_track[1] = MyFeatTrack(kp, des)
        print(f"Num feat: {len(feat_track[0].good_idxs)}")

        good_idx_map, good_idx, good = self.feature.match(des_map, des)
        feat_track[0].good_idxs = feat_track[0].good_idxs[good_idx_map]
        feat_track[1].good_idxs = feat_track[1].good_idxs[good_idx]
        print(f"Num good match: {len(feat_track[0].good_idxs)}")

        _, inliers_mask = recoverPose(feat_track[1].good_kps_n(), feat_track[0].good_kps_n())  
        # Mask inliers kps
        feat_track[0].good_idxs = feat_track[0].good_idxs[inliers_mask.ravel()==1]
        feat_track[1].good_idxs = feat_track[1].good_idxs[inliers_mask.ravel()==1]
        print(f"Num inliers: {len(feat_track[0].good_idxs)}")

        # Filter non-unique idxs
        unique_idxs = return_unique_mask(feat_track[1].good_idxs)
        feat_track[0].good_idxs = feat_track[0].good_idxs[unique_idxs]
        feat_track[1].good_idxs = feat_track[1].good_idxs[unique_idxs]
        assert len(set(feat_track[0].good_idxs)) == len(set(feat_track[1].good_idxs))
        print(f"Num unique inliers: {len(feat_track[0].good_idxs)}")

        points3d_map_matched = points3d_map[feat_track[0].good_idxs]
        map_points_matched = np.array(sfm_map.get_latest_map_points())[feat_track[0].good_idxs]
        pose_0_2 = estimate_pose_from_map_correspondences(self.K, feat_track[1].good_kps().T, points3d_map_matched.T) # pose_w_c, w == world_frame, new == cam_frame

        # Add keyframe to map
        kf = Keyframe(matched_frame, pose_0_2, rtk_pose, ts)
        kf.current_imu_pose = sfm_map.get_keyframe(kf.id()-1).current_imu_pose
        kf.current_vel = sfm_map.get_keyframe(kf.id()-1).current_vel
        kf.current_bias = sfm_map.get_keyframe(kf.id()-1).current_bias

        sfm_map.add_keyframe(kf)

        color_img = matched_frame.load_image()
        cam_ind = matched_frame.id()
        num_points = len(points3d_map_matched)
        for i,curr_map_point in enumerate(map_points_matched):
            det_id = feat_track[1].good_idxs[i]
            det_point = feat_track[1].good_kps()[i].reshape(2,1)
            curr_map_point.add_observation(kf, det_id)

            if int(det_point[1]) < color_img.shape[0] and int(det_point[1]) >= 0 and int(det_point[0]) >= 0 and int(det_point[0]) < color_img.shape[1]:
                color = np.reshape(color_img[int(det_point[1]), int(det_point[0])], (3,1))
            else:
                color = np.zeros((3,1))

            curr_track = FeatureTrack()
            matched_frame.add_keypoint(det_id, KeyPoint(det_point, color, curr_track))

        return sfm_map

    def track_map_only_imu(self, sfm_map, img_path=None, img=None, rtk_pose=None, ts=None):
        frame_idx = sfm_map._cur_keyframe_id + 1
        matched_frame = MatchedFrame(frame_idx, PerspectiveCamera(self.f, self.principal_point), img_path, img)
        kf = Keyframe(matched_frame, SE3(), rtk_pose, ts)
        kf.current_imu_pose = sfm_map.get_keyframe(kf.id()-1).current_imu_pose
        kf.current_vel = sfm_map.get_keyframe(kf.id()-1).current_vel
        kf.current_bias = sfm_map.get_keyframe(kf.id()-1).current_bias

        sfm_map.add_keyframe(kf)
        return sfm_map

    def create_new_map_points(self, sfm_map):
        print("New map points")
        print("#" * 20)

        kf_0 = sfm_map.get_keyframe(sfm_map._cur_keyframe_id - 2)
        kf_1 = sfm_map.get_keyframe(sfm_map._cur_keyframe_id)
        kfs = [kf_0, kf_1]
        matched_frames = [kf_0._frame, kf_1._frame]

        # Load images
        img0 = kf_0._frame.load_image_grayscale()
        img1 = kf_1._frame.load_image_grayscale()

        # Detect features
        kp0, des0 = self.feature.detectAndCompute(img0)
        kp1, des1 = self.feature.detectAndCompute(img1)

        feat_track = [None] * 2
        feat_track[0] = MyFeatTrack(kp0, des0)
        feat_track[1] = MyFeatTrack(kp1, des1)
        print(f"Num feat: {len(feat_track[0].good_idxs)}")

        # Match features
        good_idx0, good_idx1, _ = self.feature.match(des0, des1)
        feat_track[0].good_idxs = feat_track[0].good_idxs[good_idx0]
        feat_track[1].good_idxs = feat_track[1].good_idxs[good_idx1]
        print(f"Num good match: {len(feat_track[0].good_idxs)}")

        _, inliers_mask = recoverPose(feat_track[1].good_kps_n(), feat_track[0].good_kps_n())  

        # Mask inliers kps
        feat_track[0].good_idxs = feat_track[0].good_idxs[inliers_mask.ravel()==1]
        feat_track[1].good_idxs = feat_track[1].good_idxs[inliers_mask.ravel()==1]
        print(f"Num inliers: {len(feat_track[0].good_idxs)}")

        P_0 = kf_0._frame.camera_model().projection_matrix(kf_0.pose_w_c().inverse())
        P_1 = kf_1._frame.camera_model().projection_matrix(kf_1.pose_w_c().inverse())
        points_0 = triangulate_points_from_two_views(P_0, feat_track[0].good_kps().T, P_1, feat_track[1].good_kps().T)

        # Create 3d points with unique id
        # Add 3d points to match feat0
        points_0_obj = [UniquePoint3D(p) for p in points_0.T].copy()
        feat_track[0].set_points3d(points_0_obj.copy())
        feat_track[1].set_points3d(points_0_obj.copy())

        if FILT_DEPTH:
            ## Filter depth, using kf0 and points given in its camera coordinate
            T_c_w = kf_0.pose_w_c().inverse().to_matrix()
            points_c = (T_c_w @ add_ones(points_0.T).T)[:3,:]

            # Remove points with negative depth for kf_0
            positive_depth_idxs = points_c[2,:] > MIN_DEPTH

            # Remove long distance points
            max_depth_mask = points_c[2,:] < MAX_DEPTH

            depth_mask = np.logical_and(positive_depth_idxs, max_depth_mask)
            feat_track[0].good_idxs = feat_track[0].good_idxs[depth_mask]
            feat_track[1].good_idxs = feat_track[1].good_idxs[depth_mask]
            ##

        # Reprojection error
        pt3d_h0 = add_ones(feat_track[0].get_plot_points3d().squeeze())
        uv_hat0 = P_0 @ pt3d_h0.T
        uv_hat0 = (uv_hat0[:2]/uv_hat0[-1]).T
        uv0 = feat_track[0].good_kps()

        pt3d_h1 = add_ones(feat_track[1].get_plot_points3d().squeeze())
        uv_hat1 = P_1 @ pt3d_h1.T
        uv_hat1 = (uv_hat1[:2]/uv_hat1[-1]).T
        uv1 = feat_track[1].good_kps()

        good0 = np.linalg.norm(uv0-uv_hat0, axis=1) < MAX_PIXEL_ERR
        good1 = np.linalg.norm(uv1-uv_hat1, axis=1) < MAX_PIXEL_ERR
        good_mask = np.logical_and(good0, good1)
        feat_track[0].good_idxs = feat_track[0].good_idxs[good_mask]
        feat_track[1].good_idxs = feat_track[1].good_idxs[good_mask]
        assert len(feat_track[0].good_idxs) == len(feat_track[1].good_idxs)  # unique indices check
        print(f"Num after pixel error: {len(feat_track[0].good_idxs)}")

        # Parallax filter
        # Compute back-projected rays (unit vectors) 

        rays1 = np.dot(kf_0.pose_w_c()._rotation._matrix, add_ones(feat_track[0].good_kps_n()).T).T  # vector from keyframe origin to normalized point given in world frame
        norm_rays1 = np.linalg.norm(rays1, axis=-1, keepdims=True)                  
        rays1 /= norm_rays1                      

        rays2 = np.dot(kf_1.pose_w_c()._rotation._matrix, add_ones(feat_track[1].good_kps_n()).T).T  # vector from keyframe origin to normalized point given in world frame
        norm_rays2 = np.linalg.norm(rays2, axis=-1, keepdims=True)  
        rays2 /= norm_rays2 

        # Compute dot products of rays. a dot b = |a|*|b|*cos(angle) = 1*1*cos(angle)
        cos_parallaxs = np.sum(rays1 * rays2, axis=1)  
        # Max parallax is almost 180 degrees, and min parallax angle is almost 0 degrees
        good_cos_parallaxs = np.logical_and(cos_parallaxs > 0, cos_parallaxs < cos_max_parallax)
        feat_track[0].good_idxs = feat_track[0].good_idxs[good_cos_parallaxs]
        feat_track[1].good_idxs = feat_track[1].good_idxs[good_cos_parallaxs]
        assert len(feat_track[0].good_idxs) == len(feat_track[1].good_idxs)  # unique indices check
        print(f"Num after parallax filt: {len(feat_track[0].good_idxs)}")

        # Filter non-unique idxs
        unique_idxs = return_unique_mask(feat_track[1].good_idxs)
        feat_track[0].good_idxs = feat_track[0].good_idxs[unique_idxs]
        feat_track[1].good_idxs = feat_track[1].good_idxs[unique_idxs]
        assert len(set(feat_track[0].good_idxs)) == len(set(feat_track[1].good_idxs)), \
            f"unique indices check {len(set(feat_track[0].good_idxs))} vs {len(set(feat_track[1].good_idxs))}"
        print(f"Num unique inliers: {len(feat_track[0].good_idxs)}")


        color_img = kf_0._frame.load_image()

        latest_map_points = []
        num_points = len(feat_track[0].good_idxs)
        for i in range(num_points):
            curr_track = FeatureTrack()
            # KF_0 is the keyframe to triangulate using
            curr_map_point_id = feat_track[0].good_pts3d_w()[i].id
            curr_map_point_coord = feat_track[0].good_pts3d_w()[i].point
            curr_map_point_des = feat_track[0].good_des()[i]  # first frame is the keyframe, so add it's descriptor
            curr_map_point_kps_raw = feat_track[0].good_kps_raw()[i]  # first frame is the keyframe, so add it's keypoint
            curr_map_point = MapPoint(curr_map_point_id, curr_map_point_coord, curr_map_point_des, curr_map_point_kps_raw)

            for cam_ind in range(len(matched_frames)):
                det_id = feat_track[cam_ind].good_idxs[i]
                det_point = feat_track[cam_ind].good_kps()[i].reshape(2,1)

                if int(det_point[1]) < color_img.shape[0] and int(det_point[1]) >= 0 and int(det_point[0]) >= 0 and int(det_point[0]) < color_img.shape[1]:
                    color = np.reshape(color_img[int(det_point[1]), int(det_point[0])], (3,1))
                else:
                    color = np.zeros((3,1))

                curr_track.add_observation(matched_frames[cam_ind], det_id)
                matched_frames[cam_ind].add_keypoint(det_id, KeyPoint(det_point, color, curr_track))
                curr_map_point.add_observation(kfs[cam_ind], det_id)

            latest_map_points.append(curr_map_point)
            sfm_map.add_map_point(curr_map_point)

        sfm_map.set_latest_map_points(latest_map_points)
        return sfm_map
    
    def cull_bad_map_points(self, sfm_map):
        # Remove keyframe observations of map points
        bad_observations = {}
        for keyframe in sfm_map.get_keyframes():
            bad_keypoint_ids = []
            for keypoint_id, map_point in keyframe.get_observations().items():
                # Less than three keyframe views
                # too far away
                # behind the camera
                if map_point.num_observations() < MIN_NUM_OBS or np.linalg.norm(map_point._point_w) > MAX_DEPTH:
                    bad_keypoint_ids.append(keypoint_id)
            bad_observations[keyframe.id()] = bad_keypoint_ids
                    
        for keyframe_id in bad_observations.keys():
            for keypoint_id in bad_observations[keyframe_id]:
                sfm_map.get_keyframe(keyframe_id).remove_map_point(keypoint_id)

        # Delete points from map
        bad_map_point_ids = []
        for map_point in sfm_map.get_map_points():
            if map_point.num_observations() < MIN_NUM_OBS or np.linalg.norm(map_point._point_w) > MAX_DEPTH:
                bad_map_point_ids.append(map_point.id())
        [sfm_map.remove_map_point(map_point_id) for map_point_id in bad_map_point_ids]

        # Delete points from latest map points
        bad_map_points = []
        for map_point in sfm_map.get_latest_map_points():
            if map_point.num_observations() < MIN_NUM_OBS or np.linalg.norm(map_point._point_w) > MAX_DEPTH:
                bad_map_points.append(map_point)
        [sfm_map.remove_map_point_latest(map_point) for map_point in bad_map_points]