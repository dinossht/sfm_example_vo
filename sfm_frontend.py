from sfm_map import SfmMap, Keyframe, MapPoint, KeyPoint, MatchedFrame, PerspectiveCamera, FeatureTrack
from pylie import SO3, SE3
from frontend import *
from utils import *
from sfm_frontend_utils import *


# TODO: add undistort

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
        self.kps = convertToPoints(kps)
        self.kps_n = undistort_normalize(self.kps)

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
    def __init__(self):
        self.feature = Feature(2000)

        # Hardcoded
        self.f = np.array([[1.9954e+03, 1.9952e+03]]).T
        self.principal_point = np.array([[9.6550e+02, 6.0560e+02]]).T
        self.K = K

    def initialize(self, img0_path, img1_path):
        print("Initializing")
        print("#" * 20)
        # Init match frames
        matched_frames = [
            MatchedFrame(0, PerspectiveCamera(self.f, self.principal_point), img0_path),
            MatchedFrame(1, PerspectiveCamera(self.f, self.principal_point), img1_path)]

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
        good_idx0, good_idx1, _ = self.feature.goodMatches(des0, des1)
        feat_track[0].good_idxs = feat_track[0].good_idxs[good_idx0]
        feat_track[1].good_idxs = feat_track[1].good_idxs[good_idx1]
        print(f"Num good match: {len(feat_track[0].good_idxs)}")

        T_0_1, inliers_mask = recoverPose(feat_track[1].good_kps_n(), feat_track[0].good_kps_n())  
        pose_0_1 = SE3((SO3(T_0_1[:3,:3]), T_0_1[:3,3])) # pose_w_c, 0 == world_frame, 1 == cam_frame
        #print(pose_0_1)

        # Mask inliers kps
        feat_track[0].good_idxs = feat_track[0].good_idxs[inliers_mask.ravel()==1]
        feat_track[1].good_idxs = feat_track[1].good_idxs[inliers_mask.ravel()==1]
        print(f"Num inliers: {len(feat_track[0].good_idxs)}")

        P_0 = matched_frames[0].camera_model().projection_matrix(SE3())
        P_1 = matched_frames[1].camera_model().projection_matrix(pose_0_1.inverse())
        points_0 = triangulate_points_from_two_views(P_0, feat_track[0].good_kps().T, P_1, feat_track[1].good_kps().T)

        # Create 3d points with unique id
        # Add 3d points to match feat0
        points_0_obj = [UniquePoint3D(p) for p in points_0.T].copy()
        feat_track[0].set_points3d(points_0_obj.copy())
        feat_track[1].set_points3d(points_0_obj.copy())

        # Filter depth
        max_point_dist = 100
        depth_mask = np.logical_and(points_0[2,:]>0, points_0[2,:]<max_point_dist)
        feat_track[0].good_idxs = feat_track[0].good_idxs[depth_mask]
        feat_track[1].good_idxs = feat_track[1].good_idxs[depth_mask]

        assert len(set(feat_track[0].good_idxs)) == len(set(feat_track[1].good_idxs))  # unique indices check

        sfm_map = SfmMap()
        # Add first keyframe as reference frame.
        kf_0 = Keyframe(matched_frames[0], SE3())
        sfm_map.add_keyframe(kf_0)
        # Add second keyframe from relative pose.
        kf_1 = Keyframe(matched_frames[1], pose_0_1)
        sfm_map.add_keyframe(kf_1)

        color_img = matched_frames[0].load_image()

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

                color = np.reshape(color_img[int(det_point[1]), int(det_point[0])], (3,1))

                curr_track.add_observation(matched_frames[cam_ind], det_id)
                matched_frames[cam_ind].add_keypoint(det_id, KeyPoint(det_point, color, curr_track))
                curr_map_point.add_observation(sfm_map.get_keyframe(cam_ind), det_id)

            sfm_map.add_map_point(curr_map_point)

        return sfm_map

    def track_map(self, sfm_map, frame_idx, img_path):
        print("Tracking")
        print("#" * 20)
        matched_frame = MatchedFrame(frame_idx, PerspectiveCamera(self.f, self.principal_point), img_path)

        img = matched_frame.load_image_grayscale()
        kp, des = self.feature.detectAndCompute(img)

        # Extract map points
        des_map, kp_raw_map, points3d_map = [], [], []
        for map_point in sfm_map.get_map_points():
            des_map.append(map_point._des)
            kp_raw_map.append(map_point._kps_raw)
            points3d_map.append(map_point.point_w())

        des_map = np.array(des_map)
        kp_raw_map = np.array(kp_raw_map)
        points3d_map = np.array(points3d_map).squeeze()
        ##

        feat_track = [None] * 2
        feat_track[0] = MyFeatTrack(kp_raw_map, des_map)
        feat_track[1] = MyFeatTrack(kp, des)
        print(f"Num feat: {len(feat_track[0].good_idxs)}")

        good_idx_map, good_idx, good = self.feature.goodMatches(des_map, des)
        feat_track[0].good_idxs = feat_track[0].good_idxs[good_idx_map]
        feat_track[1].good_idxs = feat_track[1].good_idxs[good_idx]
        print(f"Num good match: {len(feat_track[0].good_idxs)}")

        _, inliers_mask = recoverPose(feat_track[1].good_kps_n(), feat_track[0].good_kps_n())  
        # Mask inliers kps
        feat_track[0].good_idxs = feat_track[0].good_idxs[inliers_mask.ravel()==1]
        feat_track[1].good_idxs = feat_track[1].good_idxs[inliers_mask.ravel()==1]
        print(f"Num inliers: {len(feat_track[0].good_idxs)}")

        assert len(set(good_idx_map)) == len(set(good_idx))  # unique indices check

        points3d_map_matched = points3d_map[feat_track[0].good_idxs]
        pose_0_2 = estimate_pose_from_map_correspondences(self.K, feat_track[1].good_kps().T, points3d_map_matched.T) # pose_w_c, w == world_frame, new == cam_frame
        #print(pose_0_2)

        # Add keyframe to map
        kf = Keyframe(matched_frame, pose_0_2)
        sfm_map.add_keyframe(kf)

        color_img = matched_frame.load_image()

        num_points = len(points3d_map_matched)
        for i,curr_map_point in enumerate(np.array(list(sfm_map.get_map_points()))[feat_track[0].good_idxs]):
            curr_track = FeatureTrack()

            cam_ind = matched_frame.id()
            det_id = feat_track[1].good_idxs[i]
            det_point = feat_track[1].good_kps()[i].reshape(2,1)
            curr_map_point.add_observation(kf, det_id)

            color = np.reshape(color_img[int(det_point[1]), int(det_point[0])], (3,1))
            #curr_track.add_observation(matched_frames[cam_ind], det_id)
            matched_frame.add_keypoint(det_id, KeyPoint(det_point, color, curr_track))

        return sfm_map

    def create_new_map_points(self, frame_0, frame_1):  # keyframe 0 and keyframe 1
        #pose is known
        #triangulate using matching
        # Load images
        matched_frames = [frame_0._frame, frame_1._frame]
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
        good_idx0, good_idx1, _ = self.feature.goodMatches(des0, des1)
        feat_track[0].good_idxs = feat_track[0].good_idxs[good_idx0]
        feat_track[1].good_idxs = feat_track[1].good_idxs[good_idx1]
        print(f"Num good match: {len(feat_track[0].good_idxs)}")

        _, inliers_mask = recoverPose(feat_track[1].good_kps_n(), feat_track[0].good_kps_n())  

        # Mask inliers kps
        feat_track[0].good_idxs = feat_track[0].good_idxs[inliers_mask.ravel()==1]
        feat_track[1].good_idxs = feat_track[1].good_idxs[inliers_mask.ravel()==1]
        print(f"Num inliers: {len(feat_track[0].good_idxs)}")

        P_0 = matched_frames[0].camera_model().projection_matrix(frame_0.pose_w_c().inverse())
        P_1 = matched_frames[1].camera_model().projection_matrix(frame_1.pose_w_c().inverse())
        points_0 = triangulate_points_from_two_views(P_0, feat_track[0].good_kps().T, P_1, feat_track[1].good_kps().T)

        # Create 3d points with unique id
        # Add 3d points to match feat0
        points_0_obj = [UniquePoint3D(p) for p in points_0.T].copy()
        feat_track[0].set_points3d(points_0_obj.copy())
        feat_track[1].set_points3d(points_0_obj.copy())

        # Filter depth
        max_point_dist = 1e10  # NOTE: OBS here
        depth_mask = np.logical_and(points_0[2,:]>0, points_0[2,:]<max_point_dist)
        feat_track[0].good_idxs = feat_track[0].good_idxs[depth_mask]
        feat_track[1].good_idxs = feat_track[1].good_idxs[depth_mask]

        assert len(set(feat_track[0].good_idxs)) == len(set(feat_track[1].good_idxs)), f"unique indices check {len(set(feat_track[0].good_idxs))} vs {len(set(feat_track[1].good_idxs))}"

        sfm_map = SfmMap()
        # Add first keyframe as reference frame.
        kf_0 = Keyframe(matched_frames[0], frame_0.pose_w_c())
        sfm_map.add_keyframe(kf_0)
        # Add second keyframe from relative pose.
        kf_1 = Keyframe(matched_frames[1], frame_1.pose_w_c())
        sfm_map.add_keyframe(kf_1)

        color_img = frame_0._frame.load_image()

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

                color = np.reshape(color_img[int(det_point[1]), int(det_point[0])], (3,1))

                curr_track.add_observation(matched_frames[cam_ind], det_id)
                matched_frames[cam_ind].add_keypoint(det_id, KeyPoint(det_point, color, curr_track))
                curr_map_point.add_observation(sfm_map.get_keyframe(cam_ind), det_id)

            sfm_map.add_map_point(curr_map_point)

        return sfm_map