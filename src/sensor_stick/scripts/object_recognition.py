#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
	get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
	return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
	# Convert ROS msg to PCL data
	pcl_data = ros_to_pcl(pcl_msg)
	
	# Voxel Grid Downsampling
	vox_filter = pcl_data.make_voxel_grid_filter()
	LEAF_SIZE = 0.01
	vox_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
	pcl_vox_filtered = vox_filter.filter()
	
	# PassThrough Filter
	passthru = pcl_vox_filtered.make_passthrough_filter()
	filtered_axis = 'z'
	passthru.set_filter_field_name(filtered_axis)
	axis_min, axis_max = 0.6, 1.1
	passthru.set_filter_limits(axis_min, axis_max)
	pcl_vox_passthru_filtered = passthru.filter()
	
	# RANSAC Plane Segmentation
	seg = pcl_vox_passthru_filtered.make_segmenter()
	seg.set_model_type(pcl.SACMODEL_PLANE)
	seg.set_method_type(pcl.SAC_RANSAC)
	max_distance = 0.01
	seg.set_distance_threshold(max_distance)
	
	# Extract inliers and outliers
	inliers, coefficients = seg.segment()
	cloud_objects = pcl_vox_passthru_filtered.extract(inliers, negative=True)
	cloud_table = pcl_vox_passthru_filtered.extract(inliers, negative=False)
	
	# Euclidean Clustering
	white_cloud = XYZRGB_to_XYZ(cloud_objects)
	tree = white_cloud.make_kdtree()
	
	# Create Cluster-Mask Point Cloud to visualize each cluster separately
	ec = white_cloud.make_EuclideanClusterExtraction()
	
	ec.set_ClusterTolerance(0.015)
	ec.set_MinClusterSize(17)
	ec.set_MaxClusterSize(1300)
	
	# Search the k-d tree for clusters
	ec.set_SearchMethod(tree)
	# Extract indices for each of the discovered clusters
	cluster_indices = ec.Extract()
	
	# Assign a color corresponding to each segmented object in scene
	cluster_color = get_color_list(len(cluster_indices))
	print(len(cluster_indices))
	color_cluster_point_list = []
	for j, indices in enumerate(cluster_indices):
		for i, indice in enumerate(indices):
			color_cluster_point_list.append([white_cloud[indice][0], white_cloud[indice][1], white_cloud[indice][2], rgb_to_float(cluster_color[j])])
	
	# Create new cloud containing all clusters, each with unique color
	cluster_cloud = pcl.PointCloud_PointXYZRGB()
	cluster_cloud.from_list(color_cluster_point_list)
	
	# Convert PCL data to ROS messages
	ros_cloud_table = pcl_to_ros(cloud_table)
	ros_cloud_objects = pcl_to_ros(cloud_objects)
	ros_cluster_cloud = pcl_to_ros(cluster_cloud)

	# Publish ROS messages
	pcl_objects_pub.publish(ros_cloud_objects)
	pcl_table_pub.publish(ros_cloud_table)
	pcl_cluster_pub.publish(ros_cluster_cloud)

	detected_objects_labels = []
	detected_objects = []
	
	for index, pts_list in enumerate(cluster_indices):
		pcl_cluster = cloud_objects.extract(pts_list)	
		ros_cluster = pcl_to_ros(pcl_cluster)
		chists = compute_color_histograms(ros_cluster, using_hsv=True)
		normals = get_normals(ros_cluster)
		nhists = compute_normal_histograms(normals)
		feature = np.concatenate((chists, nhists))
	
		# Make the prediction, retrieve the label for the result
		# and add it to detected_objects_labels list		
		prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
		label = encoder.inverse_transform(prediction)[0]
		detected_objects_labels.append(label)

		# Publish a label into RViz
		label_pos = list(white_cloud[pts_list[0]])
		label_pos[2] += .4
		object_markers_pub.publish(make_label(label,label_pos, index))

		# Add the detected object to the list of detected objects.
		do = DetectedObject()
		do.label = label
		do.cloud = ros_cluster
		detected_objects.append(do)

	rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

	# Publish the list of detected objects
	# This is the output you'll need to complete the upcoming project!
	detected_objects_pub.publish(detected_objects)
	

if __name__ == '__main__':
	# ROS node initialization
	rospy.init_node('clustering', anonymous=True)
	
	# Create Subscribers
	pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

	# Create Publishers
	pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
	pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
	pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
	object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
	detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
	# Load Model From disk
	model = pickle.load(open('model.sav', 'rb'))
	clf = model['classifier']
	encoder = LabelEncoder()
	encoder.classes_ = model['classes']
	scaler = model['scaler']

	# Initialize color_list
	get_color_list.color_list = []

	# TODO: Spin while node is not shutdown
	while not rospy.is_shutdown():
		rospy.spin()

