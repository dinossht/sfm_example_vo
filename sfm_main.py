
from optimize import BatchBundleAdjustment
from sfm_frontend import SFM_frontend
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import cv2


# TODO: use orbslam to find transformation-camera-body?
# TODO: add undistort
# TODO: remove bad map points, negative depth etc. check pyslam


i = 0
N = 10
off = 0
def next_frame_path():
    global i, N, off
    path = f"kaia_data/frame_id_{off + i * N}.png"
    frame_id = i
    i += 1
    return path, frame_id

def main():
    optimizer = BatchBundleAdjustment()

    sfm_frontend = SFM_frontend(5000, 0.7)

    # Add frame 0 and 10 to init
    path0, _ = next_frame_path()
    path1, _ = next_frame_path()
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
        """
        points3d = np.array(list(sfm_map.get_map_points()))
        points3d = np.array([p._point_w for p in points3d]).squeeze()
        plt.scatter(points3d[:,0], points3d[:,2])
        """

        for _ in range(3):
            optimizer.full_bundle_adjustment_update(sfm_map)

        """
        points3d = np.array(list(sfm_map.get_map_points()))
        points3d = np.array([p._point_w for p in points3d]).squeeze()
        plt.scatter(points3d[:,0], points3d[:,2])
        plt.show()
        """
        """
        ctr = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2021-05-30-08-56-25.json")
        ctr.convert_from_pinhole_camera_parameters(parameters)
        """

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

    # Create visualizer.
    key_to_callback = {}
    key_to_callback[ord("O")] = optimize
    key_to_callback[ord("T")] = track_new_frame
    key_to_callback[ord("C")] = create_new_points
    key_to_callback[ord("D")] = cull_bad_points
    key_to_callback[ord("I")] = iterate
    o3d.visualization.draw_geometries_with_key_callbacks(get_geometry(), key_to_callback)


if __name__ == "__main__":
    main()