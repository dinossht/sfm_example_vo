import os
import cv2
import yaml
import rospy
import rosbag
import numpy as np
import pymap3d as pm
from enum import Enum
from PIL import ImageFile

from ferryslam.parameters import Parameters as par

# TODO: Fix count num of messages

# NOTE: this file must stay in the root folder of the repository
# get the folder location of this file!
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class DatasetType(Enum):
    VIDEO = 1
    ROS = 2


def dataset_factory(dataset_type: DatasetType):
    if dataset_type == DatasetType.VIDEO:
        path = os.path.join(__location__, "videos/kitti06/video_color.mp4")
        return VideoDataset(path)
    if dataset_type == DatasetType.ROS:
        dir = os.path.join(__location__, "rosbags")
        return ROSDataset(
            dir, par.bag_file_name, par.bag_start_time_offset_s, par.bag_duration_s
        )


class ROSDataset:
    """Class for reading ros message topics"""

    def __init__(self, folderpath, filename, bag_start_time_offset=0, bag_duration=10):
        # Read ros bag
        self.filepath = os.path.join(folderpath, filename + ".bag")
        self.bag = rosbag.Bag(self.filepath)
        print("ROS bag: ", self.filepath)

        # Get bag start time
        self.bag_start_time = rospy.Time(
            self.bag.get_start_time() + bag_start_time_offset
        )
        if bag_duration == -1:
            self.bag_end_time = rospy.Time(self.bag.get_end_time())
        else:
            self.bag_end_time = rospy.Time(
                self.bag.get_start_time() + bag_start_time_offset + bag_duration
            )

        # Read camera topic and create a generator object to get the messages
        cam_topic = "/camera/image_raw/compressed"
        self.camera_freq = self.get_topic_frequency(cam_topic)
        self.num_frames = int(self.bag.get_message_count(cam_topic))
        self.cam_msgs = self.bag.read_messages(
            start_time=self.bag_start_time,
            end_time=self.bag_end_time,
            topics=[cam_topic],
        )
        print("num frames: ", self.num_frames)

        # Read imu topic and create a generator object to get the messages
        imu_topic = "/sentiboard/adis"
        self.imu_freq = self.get_topic_frequency(imu_topic)
        self.num_imu_data = int(self.bag.get_message_count(imu_topic))
        self.imu_msgs = self.bag.read_messages(
            start_time=self.bag_start_time,
            end_time=self.bag_end_time,
            topics=[imu_topic],
        )
        print("num imu data: ", self.num_imu_data)

        # Read lidar imu topic and create a generator object to get the messages
        lidar_imu_topic = "/os1_cloud_node/imu"
        self.lidar_imu_freq = self.get_topic_frequency(lidar_imu_topic)
        self.num_lidar_imu_data = int(self.bag.get_message_count(lidar_imu_topic))
        self.lidar_imu_msgs = self.bag.read_messages(
            start_time=self.bag_start_time,
            end_time=self.bag_end_time,
            topics=[lidar_imu_topic],
        )
        print("num lidar imu data: ", self.num_lidar_imu_data)

        # Read gnss topic and create a generator object to get the messages
        gnss1_topic = "/ublox1/fix"
        self.gnss1_freq = self.get_topic_frequency(gnss1_topic)
        self.num_gnss1_data = int(self.bag.get_message_count(gnss1_topic))
        self.gnss1_msgs = self.bag.read_messages(
            start_time=self.bag_start_time,
            end_time=self.bag_end_time,
            topics=[gnss1_topic],
        )
        print("num gnss1 data: ", self.num_gnss1_data)

        # Read gnss topic and create a generator object to get the messages
        gnss2_topic = "/ublox2/fix"
        self.gnss2_freq = self.get_topic_frequency(gnss2_topic)
        self.num_gnss2_data = int(self.bag.get_message_count(gnss2_topic))
        self.gnss2_msgs = self.bag.read_messages(
            start_time=self.bag_start_time,
            end_time=self.bag_end_time,
            topics=[gnss2_topic],
        )
        print("num gnss2 data: ", self.num_gnss2_data)

    def get_topic_frequency(self, topic):
        """Get frequency of a topic in bag"""
        info_dict = yaml.load(self.bag._get_yaml_info(), Loader=yaml.FullLoader)
        freq = [t["frequency"] for t in info_dict["topics"] if t["topic"] == topic][0]
        return freq

    def print_bag_topics_info(self):
        """Print ROS bag topics info"""
        info_dict = yaml.load(self.bag._get_yaml_info(), Loader=yaml.FullLoader)
        for topic in info_dict["topics"]:
            print(topic)

    def _img_from_CompressedImage(self, msg):
        """Convert compressed camera image to numpy array"""
        parser = ImageFile.Parser()
        parser.feed(msg.data)
        res = parser.close()
        return np.array(res)

    def get_image(self):
        """Get camera frame and timestamp"""
        topic, msg, t = next(self.cam_msgs)
        image = self._img_from_CompressedImage(msg)
        image = image[:, :, ::-1]  # Convert RGB to BGR

        timestamp = t.to_sec()
        frame_id = None
        return np.array(image), frame_id, timestamp

    def get_imu(self):
        """Get imu data and timestamp"""
        topic, msg, t = next(self.imu_msgs)
        timestamp = t.to_sec()
        return msg, timestamp

    def get_gnss1(self):
        """Get gnss data and timestamp"""
        topic, msg, t = next(self.gnss1_msgs)
        timestamp = t.to_sec()
        return msg, timestamp

    def get_gnss2(self):
        """Get gnss data and timestamp"""
        topic, msg, t = next(self.gnss2_msgs)
        timestamp = t.to_sec()
        return msg, timestamp

    def get_gnss_in_NED(self, gnss_type, origin_lat0, origin_lon0, origin_hei0):
        gnss_msgs = None
        if gnss_type == 1:
            gnss_msgs = self.gnss1_msgs
        if gnss_type == 2:
            gnss_msgs = self.gnss2_msgs
        gnss_msg_list = list(gnss_msgs)

        # Initialize empty arrays
        GPS_data = np.zeros((len(gnss_msg_list), 3))
        GPS_times = np.zeros((len(gnss_msg_list),))
        for i in range(len(gnss_msg_list)):
            _, gnss_msg, t = gnss_msg_list[i]
            n, e, d = pm.geodetic2ned(
                gnss_msg.latitude,
                gnss_msg.longitude,
                gnss_msg.altitude,
                origin_lat0,  # NED origin
                origin_lon0,  # NED origin
                origin_hei0,  # NED origin
                ell=pm.Ellipsoid("wgs84"),
                deg=True,
            )
            # Extract NED, altitude and time array
            GPS_data[i] = np.array([n, e, d])
            GPS_times[i] = t.to_sec()
        return GPS_data, GPS_times

    def get_imu_acc_gyro(self):
        imu_msg_list = list(self.imu_msgs)

        # Initialize empty arrays
        IMU_data = np.zeros((len(imu_msg_list), 6))
        IMU_times = np.zeros((len(imu_msg_list),))
        for i in range(len(imu_msg_list)):
            _, imu_msg, t = imu_msg_list[i]
            # Extract pose and time array
            IMU_data[i] = np.array(
                [
                    imu_msg.linear_acceleration.x,
                    imu_msg.linear_acceleration.y,
                    imu_msg.linear_acceleration.z,
                    imu_msg.angular_velocity.x,
                    imu_msg.angular_velocity.y,
                    imu_msg.angular_velocity.z,
                ]
            )
            IMU_times[i] = t.to_sec()
        return IMU_data, IMU_times

    def close(self):
        """Close bag file"""
        print("Closing bag ...")
        self.bag.close()


class VideoDataset:
    def __init__(self, path):
        self.file_path = path
        self.frame_id = -1

        self.camera_freq = 1  # image frequency

        print("video: ", self.file_path)
        self.cap = cv2.VideoCapture(self.file_path)

        if not self.cap.isOpened():
            raise IOError("Cannot open movie file: ", self.file_path)
        else:
            print("Processing Video Input")
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("num frames: ", self.num_frames)

    def get_image(self):
        """Get camera frame and timestamp"""
        ret, image = self.cap.read()
        if ret is False:
            print("ERROR while reading from file: ", self.file_path)
        self.frame_id += 1
        timestamp = None
        return image, self.frame_id, timestamp

    def close(self):
        pass
