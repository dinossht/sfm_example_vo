
from optimize import BatchBundleAdjustment
from sfm_frontend import SFM_frontend
import open3d as o3d
from video_dataset import VideoDataset
import matplotlib.pyplot as plt
import numpy as np
import cv2


# TODO: add undistort


i = 0
N = 10
def next_frame_path():
    global i, N
    path = f"kaia_data/frame_id_{i * N}.png"
    frame_id = i
    i += 1
    return path, frame_id

def main():
    optimizer = BatchBundleAdjustment()

    sfm_frontend = SFM_frontend()

    path0, _ = next_frame_path()
    path1, _ = next_frame_path()
    sfm_map = sfm_frontend.initialize(path0, path1)

    #sfm_map = sfm_frontend.create_new_map_points(sfm_map.get_keyframe(2), sfm_map.get_keyframe(3))

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
        # TODO: cull bad points

        points3d = np.array(list(sfm_map.get_map_points()))
        points3d = np.array([p._point_w for p in points3d]).squeeze()
        plt.scatter(points3d[:,0], points3d[:,2])
        plt.show()


        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)

    def add_new_frame(vis):
            path, frame_id = next_frame_path()
            sfm_frontend.track_map(sfm_map, frame_id, path)

            vis.clear_geometries()
            for geom in get_geometry():
                vis.add_geometry(geom, reset_bounding_box=False)

    # Create visualizer.
    key_to_callback = {}
    key_to_callback[ord("O")] = optimize
    key_to_callback[ord("A")] = add_new_frame
    o3d.visualization.draw_geometries_with_key_callbacks(get_geometry(), key_to_callback)


if __name__ == "__main__":
    main()