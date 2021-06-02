# %%
import numpy as np
from sensor_dataset import ROSDataset
from ground_truth import ROSGroundTruth
import matplotlib.pyplot as plt
from utils import eulers_from_pose_in_deg
# %%

trip_nr = 3
t_start = 650
dat = ROSDataset("dataset/rosbags", f"trondheim{trip_nr}_inn", t_start, -1)
gt = ROSGroundTruth(f"dataset/groundtruths/obsv_estimates ({trip_nr}).mat", "dataset/groundtruths/ned_origin.mat", trip_nr)

# %%
img, _, first_timestamp = dat.get_image()

T_bodys = []
T_cams = []
for i in range(80):
    T_body = gt.get_T_body(first_timestamp + i)
    T_bodys.append(T_body)

    T_cam = gt.get_T_cam(first_timestamp + i)
    T_cams.append(T_cam)

T_bodys = np.array(T_bodys)
T_cams = np.array(T_cams)
%matplotlib qt

# %%
plt.plot(T_bodys[:,1,3], T_body[:,0,3])
plt.plot(T_cams[:,1,3], T_cams[:,0,3])
plt.show

# %%
plt.plot(T_bodys[:,0,3], T_bodys[:,2,3])
plt.plot(T_cams[:,0,3], T_cams[:,2,3])
plt.show()
# %%
plt.plot(T_bodys[:,0,3])
plt.plot(T_cams[:,0,3])
plt.show()

# %%
eulers_body = eulers_from_pose_in_deg(T_bodys)
eulers_cam = eulers_from_pose_in_deg(T_cams)
plt.plot(eulers_body[:,2])
plt.plot(eulers_cam[:,1])
plt.legend(["b", "c"])

plt.show()

# %%
