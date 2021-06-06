import numpy as np
import gtsam
from gtsam import PriorFactorPoint3
from sfm_map import SfmMap
from pylie import SE3, SO3
from utils import *
import matplotlib.pyplot as plt
from parameters import param


ADD_RTK_PRIOR = False

ADD_GPS_PRIOR = False
ADD_IMU_FACTOR = True
ADD_CAMERA_FACTOR = True

# NOTE: for IMU + GPS use inv_sigma = 1 and not 100
inv_sigma_rtk = 100
inv_sigma_gps = 1
imu_scale = 1

if ADD_CAMERA_FACTOR:
    inv_sigma_gps = 100
    imu_scale = 1
if ADD_CAMERA_FACTOR and ADD_IMU_FACTOR and not ADD_GPS_PRIOR:
    imu_scale = 0.1



"""Setup IMU preintegration and bias parameters"""
AccSigma        = 0.01 * imu_scale
GyroSigma       = 0.000175 * imu_scale
IntSigma        = 0.000167 * imu_scale # integtation sigma
AccBiasSigma    = 2.91e-006 * imu_scale
GyroBiasSigma   = 0.0100395199348279 * imu_scale
preintegration_param = gtsam.PreintegrationParams(np.array([0, 0, 9.82175]))
preintegration_param.setAccelerometerCovariance(AccSigma ** 2 * np.eye(3))
preintegration_param.setGyroscopeCovariance(GyroSigma ** 2 * np.eye(3))
preintegration_param.setIntegrationCovariance(IntSigma ** 2 * np.eye(3))
preintegration_param.setOmegaCoriolis(np.array([0, 0, 0]))  # account for earth's rotation
sigmaBetweenBias = np.array([AccBiasSigma, AccBiasSigma, AccBiasSigma, GyroBiasSigma, GyroBiasSigma, GyroBiasSigma])

class BatchBundleAdjustment:
    def full_bundle_adjustment_update(self, sfm_map: SfmMap):
        # Variable symbols for camera poses.
        X = gtsam.symbol_shorthand.X
        # Variable symbols for observed 3D point "landmarks".
        L = gtsam.symbol_shorthand.L
        # Variable symbols for IMU velocity
        V = gtsam.symbol_shorthand.V
        # Variable symbols for IMU bias
        B = gtsam.symbol_shorthand.B

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
        obs_uncertainty = gtsam.noiseModel.Isotropic.Sigma(2, 3.0)

        # The pose of the sensor in the body frame
        # body_P_sensor is the transform from body to sensor frame (default identity)
        # body_P_sensor here T_b_c, see epipolar geometry lecture TTK21 Lecture 2 
        tB_b_c = np.array([3.285, 0, -1.4])  # camera origo given in body frame
        R_xyz_zxy = np.array([[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]])
        R_b_c = R_z(13) @ R_y(-1.75) @ R_xyz_zxy
        T_b_c = np.eye(4)
        T_b_c[:3,:3] = R_b_c
        T_b_c[:3,3] = tB_b_c
        body_P_sensor_cam = gtsam.Pose3(gtsam.Rot3(R_b_c), tB_b_c)
        T_c_b = np.linalg.inv(T_b_c)

        if ADD_CAMERA_FACTOR:
            # Add measurements.
            for keyframe in sfm_map.get_keyframes():
                for keypoint_id, map_point in keyframe.get_observations().items():
                    assert len(map_point.get_observations()) >= 2, "Less than two camera views"
                    obs_point = keyframe.get_keypoint(keypoint_id).point()
                    factor = gtsam.GenericProjectionFactorCal3_S2(obs_point, obs_uncertainty,
                                                                X(keyframe.id()), L(map_point.id()),
                                                                calibration,
                                                                body_P_sensor_cam)
                    error = uv_to_X_error(obs_point.T, K, keyframe.pose_w_c().inverse(), map_point.point_w())
                    #assert error < 50, str(error)
                    graph.push_back(factor)

        # Set prior on the first camera (which we will assume defines the reference frame).
        # NOTE: prior is for the pose in body frame
        no_uncertainty_in_pose = gtsam.noiseModel.Constrained.All(6)
        kf0_T_w_b = kf_0.rtk_pose.copy()
        kf0_R_w_b = kf0_T_w_b[:3,:3]
        kf0_tW_w_b = kf0_T_w_b[:3,3]

        # USE gps z-measurement
        if ADD_GPS_PRIOR and not ADD_RTK_PRIOR:
            gnss_idx = np.argmin(np.abs(sfm_map.GNSS_times - kf_0.ts))
            prior_pos = sfm_map.GNSS_data[gnss_idx,:]
            kf0_tW_w_b[2] = prior_pos[2]

        kf0_pose_w_b = gtsam.Pose3(gtsam.Rot3(kf0_R_w_b), kf0_tW_w_b)
            
        factor = gtsam.PriorFactorPose3(X(kf_0.id()), kf0_pose_w_b, no_uncertainty_in_pose)
        graph.push_back(factor)
        
        # Set prior on the second camera
        # NOTE: prior is for the pose in body frame
        no_uncertainty_in_pose = gtsam.noiseModel.Constrained.All(6)

        kf1_T_w_b = kf_1.rtk_pose.copy()
        kf1_R_w_b = kf1_T_w_b[:3,:3]
        kf1_tW_w_b = kf1_T_w_b[:3,3]

        # USE gps z-measurement
        if ADD_GPS_PRIOR and not ADD_RTK_PRIOR:
            gnss_idx = np.argmin(np.abs(sfm_map.GNSS_times - kf_1.ts))
            prior_pos = sfm_map.GNSS_data[gnss_idx,:]
            kf1_tW_w_b[2] = prior_pos[2]

        kf1_pose_w_b = gtsam.Pose3(gtsam.Rot3(kf1_R_w_b), kf1_tW_w_b)
        factor = gtsam.PriorFactorPose3(X(kf_1.id()), kf1_pose_w_b, no_uncertainty_in_pose)
        graph.push_back(factor)

        # Add position prior RTK
        if ADD_RTK_PRIOR:
            for keyframe in sfm_map.get_keyframes():
                uncertainty_in_pos = gtsam.noiseModel.Diagonal.Precisions(np.array([0.0, 0.0, 0.0, inv_sigma_rtk, inv_sigma_rtk, inv_sigma_rtk]))
                prior_pos = keyframe.rtk_pose[:3,3]
                prior_pose = gtsam.Pose3(gtsam.Rot3(), prior_pos)
                factor = gtsam.PriorFactorPose3(X(keyframe.id()), prior_pose, uncertainty_in_pos)
                graph.push_back(factor)

        # Add position prior
        if ADD_GPS_PRIOR:
            for keyframe in sfm_map.get_keyframes():
                uncertainty_in_pos = gtsam.noiseModel.Diagonal.Precisions(np.array([0.0, 0.0, 0.0, inv_sigma_gps, inv_sigma_gps, 0.5 * inv_sigma_gps]))

                time_diff =  np.min(np.abs(sfm_map.GNSS_times - keyframe.ts)) 
                #assert time_diff < 0.05, f"large time diff gps vs keyframe. timediff: {time_diff}"
                if time_diff < 0.05:
                    gnss_idx = np.argmin(np.abs(sfm_map.GNSS_times - keyframe.ts))
                    prior_pos = sfm_map.GNSS_data[gnss_idx,:]
                    prior_pose = gtsam.Pose3(gtsam.Rot3(), prior_pos)
                    factor = gtsam.PriorFactorPose3(X(keyframe.id()), prior_pose, uncertainty_in_pos)
                    graph.push_back(factor)

        # Set initial estimates from map.
        initial_estimate = gtsam.Values()

        if ADD_IMU_FACTOR:
            # Add IMU priors
            for i, keyframe in enumerate(sfm_map.get_keyframes()):
                ts_cur = keyframe.ts

                if i == 0:
                    # Create initial estimate and prior on initial pose, velocity, and biases
                    #sigma_init_x = gtsam.noiseModel.Isotropic.Precisions([0.0, 0.0, 0.0, 1e-5, 1e-5, 1e-5])  # R, T
                    sigma_init_v = gtsam.noiseModel.Isotropic.Sigma(3, 1000.0)
                    sigma_init_b = gtsam.noiseModel.Isotropic.Sigmas([0.1, 0.1, 0.1, 5e-05, 5e-05, 5e-05])  # acc, gyro

                    initial_estimate.insert(V(keyframe.id()), keyframe.current_vel)
                    initial_estimate.insert(B(keyframe.id()), keyframe.current_bias)

                    factor_vel = gtsam.PriorFactorVector(V(keyframe.id()), keyframe.current_vel, sigma_init_v)
                    factor_bias = gtsam.PriorFactorConstantBias(B(keyframe.id()), keyframe.current_bias, sigma_init_b)

                    graph.push_back(factor_vel)
                    graph.push_back(factor_bias)

                else:
                    ## Summarize IMU data between the previous GNSS measurement and current GNSS measurement
                    IMUindices = np.argwhere((sfm_map.IMU_times >= ts_prev) & (sfm_map.IMU_times <= ts_cur)).squeeze()
                    currentSummarizedMeasurement = gtsam.PreintegratedImuMeasurements(preintegration_param, keyframe.current_bias)
                    for imuIndex in IMUindices:
                        accMeas = sfm_map.IMU_data[imuIndex, :3]
                        omegaMeas = sfm_map.IMU_data[imuIndex, 3:]
                        deltaT = 1 / 250  # imu freq
                        currentSummarizedMeasurement.integrateMeasurement(accMeas, omegaMeas, deltaT)
                    
                    ## Create IMU factor
                    factor_imu = gtsam.ImuFactor(
                        X(keyframe.id()-1),
                        V(keyframe.id()-1), 
                        X(keyframe.id()),
                        V(keyframe.id()), 
                        B(keyframe.id()-1), 
                        currentSummarizedMeasurement
                    )
                    graph.push_back(factor_imu)

                    ## Bias evolution as given in the IMU metadata
                    factor_bias = gtsam.BetweenFactorConstantBias(
                        B(keyframe.id()-1), 
                        B(keyframe.id()), 
                        gtsam.imuBias.ConstantBias(np.zeros((3, 1)), np.zeros((3, 1))),
                        gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(len(IMUindices)) * sigmaBetweenBias)
                    )
                    graph.push_back(factor_bias)

                    ## Add initial values
                    initial_estimate.insert(V(keyframe.id()), keyframe.current_vel)
                    initial_estimate.insert(B(keyframe.id()), keyframe.current_bias)

                ts_prev = ts_cur

        if ADD_CAMERA_FACTOR:
            for keyframe in sfm_map.get_keyframes():

                # Calculate pose of body frame given in world frame using camera pose (pose_w_c)
                # NOTE: prior is for the pose in body frame
                kf_R_w_c = keyframe.pose_w_c()._rotation._matrix
                kf_tW_w_c = keyframe.pose_w_c()._translation.squeeze()

                kf_T_w_c = np.eye(4)
                kf_T_w_c[:3,:3] = kf_R_w_c
                kf_T_w_c[:3,3] = kf_tW_w_c

                kf_T_w_b = kf_T_w_c @ T_c_b
                kf_R_w_b = kf_T_w_b[:3,:3]
                kf_tW_w_b = kf_T_w_b[:3,3]

                kf_pose_w_b = gtsam.Pose3(gtsam.Rot3(kf_R_w_b), kf_tW_w_b)
                initial_estimate.insert(X(keyframe.id()), kf_pose_w_b)
        
            for map_point in sfm_map.get_map_points():
                point_w = gtsam.Point3(map_point.point_w().squeeze())
                initial_estimate.insert(L(map_point.id()), point_w)
        
        if not ADD_CAMERA_FACTOR:
            for keyframe in sfm_map.get_keyframes():
                initial_estimate.insert(X(keyframe.id()), keyframe.current_imu_pose)

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
            # Update map with kamera results
            updated_pose_w_b = result.atPose3(X(keyframe.id())).matrix()
            updated_pose_w_c = updated_pose_w_b @ T_b_c

            keyframe.update_pose_w_c(SE3.from_matrix(updated_pose_w_c))

            if ADD_IMU_FACTOR:
                if not ADD_CAMERA_FACTOR:
                    keyframe.current_imu_pose = result.atPose3(X(keyframe.id()))
                keyframe.current_vel = result.atVector(V(keyframe.id()))
                keyframe.current_bias = result.atConstantBias(B(keyframe.id()))

        if ADD_CAMERA_FACTOR:
            for map_point in sfm_map.get_map_points():
                updated_point_w = result.atPoint3(L(map_point.id())).reshape(3, 1)
                map_point.update_point_w(updated_point_w)

        sfm_map.has_been_updated()