
from optimize import BatchBundleAdjustment
from sfm_frontend import SFM_frontend
from gtsam import Marginals
import open3d as o3d
from gtsam.utils import plot
import matplotlib.pyplot as plt
import numpy as np
import cv2
from load_ros_camera_rtk import camRtkData


# TODO: add prior factor for pose based on constant vel motion model
# TODO: add smart projection factor/smart factor to solve bad map points
# TODO: add parallax angle check

# NOTE: To run, click I on keyboard

# TODO: use orbslam to find transformation-camera-body?
# TODO: remove bad map points, negative depth etc. check pyslam
# TODO: plott feature matching


N = 10

dat = camRtkData(640)
def next_frame():
    img_out, T_out = dat.get_img_rtk_pos_in_IMU()
    for _ in range(N):
        img, T = dat.get_img_rtk_pos_in_IMU()
    return img_out, T_out

def main():
    optimizer = BatchBundleAdjustment()

    sfm_frontend = SFM_frontend(10000, 0.7)

    img0, T0 = next_frame()
    img1, T1 = next_frame()
    sfm_map = sfm_frontend.initialize(img0=img0, img1=img1, rtk_pos0=T0[:3,3], rtk_pos1=T1[:3,3])

    # Track a new frame
    img, T = next_frame()
    sfm_frontend.track_map(sfm_map, img=img, rtk_pos=T[:3,3])

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
        for _ in range(3):
            optimizer.full_bundle_adjustment_update(sfm_map)

        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)

    def track_new_frame(vis):
        img, T = next_frame()
        sfm_frontend.track_map(sfm_map, img=img, rtk_pos=T[:3,3])

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
        sfm_frontend.track_map(sfm_map, img=img, rtk_pos=T[:3,3])
        img, T = next_frame()
        sfm_frontend.track_map(sfm_map, img=img, rtk_pos=T[:3,3])

        # Optimize
        for j in range(5):
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

    # Create visualizer.
    key_to_callback = {}
    key_to_callback[ord("O")] = optimize
    key_to_callback[ord("T")] = track_new_frame
    key_to_callback[ord("C")] = create_new_points
    key_to_callback[ord("D")] = cull_bad_points
    key_to_callback[ord("I")] = iterate
    key_to_callback[ord("M")] = M_init_cam
    o3d.visualization.draw_geometries_with_key_callbacks(get_geometry(), key_to_callback)


if __name__ == "__main__":
    main()