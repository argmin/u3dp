## Project: Perception Pick & Place
### Abstract: We describe the methods learnt and techniques used in object recognition for robot pereption.

---

#### Problem Setup
In this project we focus on perception principle of robotics, the sensor used in this project is RGBD camera that produces point cloud data. Our goal is identify each object and position the PR2 robot to pick and drop the objects in the dropbox. In order to acheive this we perform various feature extraction methods like downsampling, filtering, segmentation, clustering, & classification for object recognition. Here each exercise build upon the past to complete this task.

#### Exercise 1
This exercise introduces us to the feature extraction from an image. Here we begin applying following steps
- Downsample 3D point cloud data voxel grid downsampling
- Perform pass through filtering for the output from previous steps along the axis to keep relevant data. Note that in the project we perform filtering along `y` & `z` axes, to filter out edge of the basket, thanks to Udacity Slack Community.
- We now perform plane fitting to previous output to inliers (table) and outlier (objects)


#### Exercise 2
In this exercise for the objects we have detected we want to identify the features of the each object, so that it can be used in object recognition. This is done clustering which in our case exploits the color to group the features of an object together. The parameters that we choose are `cluster tolerance`, `Min & Max` cluster size. These values were obtained by trial and error.

#### Exercise 3

- In this exercise we spin up the project in training mode and capture the features like color and normal histogram for an object in various orientations.
- We split the data into training and test set by random sample and train the classifier
- We evaluate the model based on confusion matrix & accuracy.


#### Project
Here we put together the steps from Exercise 1 to 3 to perform object recognition and drop the objects into right basket.
- We first spin up the ROS simulation in training mode and present the objects in this project in various orientations and collect large sample of trainign data.
- We split the data into training and test set by random sample and train the classifier
- We now launch PR2 robot in `pick_place_project` mode, extract the features as described in `Exercise 1 to 3` and classify the objects.
- After classification we find the centroid for each object and describe the output in `output_1.yaml`, `output_2.yaml`output_3.yaml` for senarios `test1.world`, `test2.world`, `test3.world`.
- In addition the pick and place location on each object is sent to robot to perform the pick and place task.


#### Results
Following are the results from training and evaluating classifier, & confusion matrix images are in the repo.

```sh
Features in Training Set: 800
Invalid Features in Training set: 4
Scores: [ 0.9375      0.94339623  0.94968553  0.94968553  0.9245283 ]
Accuracy: 0.94 (+/- 0.02)
accuracy score: 0.940954773869
````

### Code
Here's the code of interest
- `src/RoboND-Perception-Project/pr2_robot/scripts/train_svm.py`
- `src/RoboND-Perception-Project/pr2_robot/scripts/perception_pr2.py`
- `src/RoboND-Perception-Project/pr2_robot/scripts/capture_features.py`

