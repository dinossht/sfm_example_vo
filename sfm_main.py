
from optimize import BatchBundleAdjustment
from sfm_frontend import SFM_frontend
from gtsam import Marginals
import open3d as o3d
from gtsam.utils import plot
import matplotlib.pyplot as plt
import numpy as np
import gtsam
import cv2
from load_ros_camera_rtk import camRtkData


# TODO: add prior factor for pose based on constant vel motion model
# TODO: add smart projection factor/smart factor to solve bad map points

# TODO: remove bad map points, negative depth etc. check pyslam
# TODO: plott feature matching


N = 20

dat = camRtkData(650)
def next_frame():
    img_out, T_out = dat.get_img_and_rtk_pose_of_body_in_ned()
    for _ in range(N):
        img, T = dat.get_img_and_rtk_pose_of_body_in_ned()
    return img_out, T_out

def main():
    optimizer = BatchBundleAdjustment()

    sfm_frontend = SFM_frontend(10000, 0.7)

    img0, T0 = next_frame()
    img1, T1 = next_frame()
    sfm_map = sfm_frontend.initialize(img0=img0, img1=img1, rtk_pose0=T0, rtk_pose1=T1)

    # Track a new frame
    img, T = next_frame()
    sfm_frontend.track_map(sfm_map, img=img, rtk_pose=T)
    optimizer.full_bundle_adjustment_update(sfm_map)

    def get_geometry():
        poses = sfm_map.get_keyframe_poses()
        p, c = sfm_map.get_pointcloud()

        axes = []
        for pose in poses:
            axes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0).transform(pose.to_matrix()))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p.T)
        pcd.colors = o3d.utility.Vector3dVector(c.T / 255)

        return [pcd] + axes

    def optimize(vis):
        for _ in range(1):
            optimizer.full_bundle_adjustment_update(sfm_map)

        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)

    def track_new_frame(vis):
        img, T = next_frame()
        sfm_frontend.track_map(sfm_map, img=img, rtk_pose=T)

        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)

    def create_new_points(vis):
        sfm_frontend.create_new_map_points(sfm_map) # last two frames, Not last two!!!

        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)

    def cull_bad_points(vis):
        sfm_frontend.cull_bad_map_points(sfm_map)

        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)
    
    def iterate(vis):
        # Create->track->track->optimize->cull

        # Create
        sfm_frontend.create_new_map_points(sfm_map) 

        #Track Track 
        img, T = next_frame()
        sfm_frontend.track_map(sfm_map, img=img, rtk_pose=T)
        #img, T = next_frame()
        #sfm_frontend.track_map(sfm_map, img=img, rtk_pose=T)

        # Optimize
        for j in range(1):
            optimizer.full_bundle_adjustment_update(sfm_map)

        # Cull
        sfm_frontend.cull_bad_map_points(sfm_map)

        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)

    def M_init_cam(vis):
        """
        ctr = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2021-05-30-14-28-06.json")
        ctr.convert_from_pinhole_camera_parameters(parameters)
        """
        if sfm_map.graph is not None:
            # Plot trajectory and marginals
            #marginals = Marginals(sfm_map.graph, sfm_map.result)
            #plot.plot_3d_points(1, result, marginals=marginals)
            plot.plot_trajectory(1, sfm_map.result)#, marginals=marginals, scale=8)
            plot.set_axes_equal(1)        
            plt.figure()
            img = sfm_map.get_keyframe(sfm_map._cur_keyframe_id)._frame.load_image()
            plt.imshow(img)
            plt.show()

    def plot_2d_trajectory(vis):
        poses = gtsam.utilities.allPose3s(sfm_map.result)
        pos_est_ned_arr, pos_gt_ned_arr = [], []
        for key, keyframe in zip(poses.keys(), sfm_map.get_keyframes()):
            pose = poses.atPose3(key)
            pos_est_ned = pose.translation()
            pos_gt_ned = keyframe.rtk_pose[:3,3]
            pos_est_ned_arr.append(pos_est_ned)
            pos_gt_ned_arr.append(pos_gt_ned)
        pos_est_ned_arr, pos_gt_ned_arr = np.array(pos_est_ned_arr), np.array(pos_gt_ned_arr)

        plt.subplot(311)
        plt.plot(pos_est_ned_arr[:, 1], pos_est_ned_arr[:, 0], color="blue")
        plt.plot(pos_gt_ned_arr[:, 1], pos_gt_ned_arr[:, 0], color="black", linestyle="dashed")
        plt.legend(["estimate", "rtk gt"])
        plt.xlabel("y [m]")
        plt.ylabel("x [m]")

        plt.subplot(312)
        norm_xy= np.linalg.norm(pos_est_ned_arr[:, :1] - pos_gt_ned_arr[:, :1], axis=1, ord=1) # L1 norm
        plt.plot(norm_xy)
        plt.ylim([-0.1, 1])
        plt.ylabel("xy L1 norm error [m]")
        plt.legend(["L1 norm xy"])

        plt.subplot(313)
        plt.plot(pos_est_ned_arr[:, 2], color="blue")
        plt.plot(pos_gt_ned_arr[:, 2], color="black", linestyle="dashed")
        plt.ylabel("z")
        plt.xlabel("frame id")
        plt.ylim([0.5, 1.5])

        plt.suptitle("Trajectory in NED")
        plt.show()
        

    # Create visualizer.
    key_to_callback = {}
    key_to_callback[ord("O")] = optimize
    key_to_callback[ord("T")] = track_new_frame
    key_to_callback[ord("C")] = create_new_points
    key_to_callback[ord("D")] = cull_bad_points
    key_to_callback[ord("I")] = iterate
    key_to_callback[ord("M")] = M_init_cam
    key_to_callback[ord("P")] = plot_2d_trajectory
    o3d.visualization.draw_geometries_with_key_callbacks(get_geometry(), key_to_callback)


if __name__ == "__main__":
    main()