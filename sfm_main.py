
from optimize import BatchBundleAdjustment
from sfm_frontend import SFM_frontend
from gtsam import Marginals
import open3d as o3d
from gtsam.utils import plot
import matplotlib.pyplot as plt
import numpy as np
import cv2


# TODO: add parallax angle check
# NOTE: To run, click I on keyboard

# TODO: use orbslam to find transformation-camera-body?
# TODO: add undistort
# TODO: remove bad map points, negative depth etc. check pyslam
# TODO: plott feature matching


i = 0
N = 10
off = 600
def next_frame_path():
    global i, N, off
    path = f"kaia_data/frame_id_{off + i * N}.png"
    frame_id = i
    i += 1
    return path, frame_id

def main():
    optimizer = BatchBundleAdjustment()

    sfm_frontend = SFM_frontend(5000, 0.7)

    # Add frame 0 and 1 to init
    path0, _ = next_frame_path()
    path1, _ = next_frame_path()
    #path0 = "car_data/img0.png"
    #path1 = "car_data/img1.png"
    sfm_map = sfm_frontend.initialize(path0, path1)

    # Track a new frame
    path, frame_id = next_frame_path()
    sfm_frontend.track_map(sfm_map, frame_id, path)

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
        path, frame_id = next_frame_path()
        sfm_frontend.track_map(sfm_map, frame_id, path)

        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)

    def create_new_points(vis):
        sfm_frontend.create_new_map_points(sfm_map, i-3, i-1) # last two frames, Not last two!!!

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
        sfm_frontend.create_new_map_points(sfm_map, i-3, i-1) 

        #Track Track 
        path, frame_id = next_frame_path()
        sfm_frontend.track_map(sfm_map, frame_id, path)

        path, frame_id = next_frame_path()
        sfm_frontend.track_map(sfm_map, frame_id, path)

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
            marginals = Marginals(sfm_map.graph, sfm_map.result)
            #plot.plot_3d_points(1, result, marginals=marginals)
            plot.plot_trajectory(1, sfm_map.result)#, marginals=marginals, scale=8)
            plot.set_axes_equal(1)        
            plt.figure()
            last_path = sfm_map.get_keyframe(1)._frame._img_path
            plt.imshow(cv2.cvtColor(cv2.imread(last_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))
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