from sensor_dataset import ROSDataset
from ground_truth import ROSGroundTruth


trip_nr = 3
dat = ROSDataset("dataset/rosbags", f"trondheim{trip_nr}_inn", 650, -1)
gt = ROSGroundTruth(f"dataset/groundtruths/obsv_estimates ({trip_nr}).mat", "dataset/groundtruths/ned_origin.mat", trip_nr)


img, _, timestamp = dat.get_image()
g = gt.get_xyz(timestamp)
print(g.x)

img, _, timestamp = dat.get_image()
g = gt.get_xyz(timestamp)
print(g.x)