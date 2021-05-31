import numpy as np
import gtsam
from gtsam import PriorFactorPoint3
from sfm_map import SfmMap
from pylie import SE3, SO3
from utils import *
import matplotlib.pyplot as plt
from parameters import param


class BatchBundleAdjustment:
    def full_bundle_adjustment_update(self, sfm_map: SfmMap):
        # Variable symbols for camera poses.
        X = gtsam.symbol_shorthand.X

        # Variable symbols for observed 3D point "landmarks".
        L = gtsam.symbol_shorthand.L

        # Create a factor graph.
        graph = gtsam.NonlinearFactorGraph()

        # Extract the first two keyframes (which we will assume has ids 0 and 1).
        kf_0 = sfm_map.get_keyframe(0)
        kf_1 = sfm_map.get_keyframe(1)

        # We will in this function assume that all keyframes have a common, given (constant) camera model.
        common_model = kf_0.camera_model()
        calibration = gtsam.Cal3_S2(common_model.fx(), common_model.fy(), 0, common_model.cx(), common_model.cy())

        # Unfortunately, the dataset does not contain any information on uncertainty in the observations,
        # so we will assume a common observation uncertainty of 3 pixels in u and v directions.
        obs_uncertainty = gtsam.noiseModel.Isotropic.Sigma(2, 3.0) #(2, 3.0)

        

        """
        # Add measurements.
        for keyframe in sfm_map.get_keyframes():
            for keypoint_id, map_point in keyframe.get_observations().items():
                assert len(map_point.get_observations()) >= 2, "Less than two camera views"
                obs_point = keyframe.get_keypoint(keypoint_id).point()
                #factor = gtsam.GenericProjectionFactorCal3_S2(obs_point, obs_uncertainty,
                #                                              X(keyframe.id()), L(map_point.id()),
                #                                              calibration)
                smart_factor = gtsam.SmartProjectionPose3Factor(obs_uncertainty, calibration)
                smart_factor.add(obs_point, X(keyframe.id()))
                #error = uv_to_X_error(obs_point.T, K, keyframe.pose_w_c().inverse(), map_point.point_w())
                #assert error < 50, str(error)
                #graph.push_back(factor)
                graph.push_back(smart_factor)
        """

        # Add measurements.
        for map_point in sfm_map.get_map_points():
            smart_factor = gtsam.SmartProjectionPose3Factor(obs_uncertainty, calibration)
            for keyframe in map_point.get_observations():
                keypoint_id = map_point.get_observation_keypoint_id(keyframe)
                obs_point = keyframe.get_keypoint(keypoint_id).point()
                smart_factor.add(obs_point, X(keyframe.id()))
                graph.push_back(smart_factor)
                
        # Set prior on the first camera (which we will assume defines the reference frame).
        no_uncertainty_in_pose = gtsam.noiseModel.Constrained.All(6)
        #factor = gtsam.PriorFactorPose3(X(kf_0.id()), gtsam.Pose3(), no_uncertainty_in_pose)
        kf0_R = kf_0.pose_w_c()._rotation._matrix
        kf0_t = kf_0.pose_w_c()._translation
        kf0_pose = gtsam.Pose3(gtsam.Rot3(kf0_R), np.reshape(kf0_t,(3,1)))
        factor = gtsam.PriorFactorPose3(X(kf_0.id()), kf0_pose, no_uncertainty_in_pose)
        graph.push_back(factor)
        
        # Set prior on distance to next camera.
        no_uncertainty_in_distance = gtsam.noiseModel.Constrained.All(1)
        prior_distance = param.VO_SCALE
        factor = gtsam.RangeFactorPose3(X(kf_0.id()), X(kf_1.id()), prior_distance, no_uncertainty_in_distance)
        graph.push_back(factor)

        # Set initial estimates from map.
        initial_estimate = gtsam.Values()
        for keyframe in sfm_map.get_keyframes():
            pose_w_c = gtsam.Pose3(keyframe.pose_w_c().to_matrix())
            initial_estimate.insert(X(keyframe.id()), pose_w_c)

        """
        for map_point in sfm_map.get_map_points():
            point_w = gtsam.Point3(map_point.point_w().squeeze())
            initial_estimate.insert(L(map_point.id()), point_w)
        """

        # Optimize the graph.
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity('TERMINATION')
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        print('Optimizing:')
        result = optimizer.optimize()
        print('initial error = {}'.format(graph.error(initial_estimate)))
        print('final error = {}'.format(graph.error(result)))
        sfm_map.graph = graph
        sfm_map.result = result

        # Update map with results.
        for keyframe in sfm_map.get_keyframes():
            updated_pose_w_c = result.atPose3(X(keyframe.id()))
            keyframe.update_pose_w_c(SE3.from_matrix(updated_pose_w_c.matrix()))

        """
        for map_point in sfm_map.get_map_points():
            updated_point_w = result.atPoint3(L(map_point.id())).reshape(3, 1)
            map_point.update_point_w(updated_point_w)
        """

        sfm_map.has_been_updated()


class IncrementalBundleAdjustment:
    def __init__(self):
        self._isam = gtsam.ISAM2

    def full_bundle_adjustment_update(sfm_map: SfmMap):
        # You can get the new updates with.
        new_kfs = sfm_map.get_new_keyframes_not_optimized()
        new_mps = sfm_map.get_new_map_points_not_optimized()

        # Create factor graph from updates.

        # Update isam2.

        # Remember to update the results!
        sfm_map.has_been_updated()
