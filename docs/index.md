# Welcome to my Docs

This is a documentation of code for my thesis project:
<p align="center" >
    <b>"3D point cloud analysis for traffic situational awareness"</b>
</p>

The project consisted of creating an algorithm that will perform detection of dangerous situations on the road involving pedestrians. The creation of the project was accompanied by several stages:

* _Getting to the LiDAR data_ - This was my first contact with any LiDAR. I was provided with data in the old hardware and firmware version. This required physically changing the contents of the metadata file.
* _Conversion to matrix representation_ - A point cloud is an unstructured data type. For this reason, a conversion to an array representation and scaling was performed to use the detector.
* _Creation of a dataset_ - I have created dataset of data converted to image by using [Roboflow](https://roboflow.com/). The dataset is available [here](https://app.roboflow.com/lidar-object-detection).
* _Object detection_ - For this problem, I have used [YOLOv8](https://docs.ultralytics.com/) convolution neural network. I have trained it on my dataset.
* _Object tracking_ - For this problem, I have also used [YOLOv8](https://docs.ultralytics.com/) with `ByteTrack` tracker.
* _Creation of my own algorithm_ - This stage included selecting the point representing the detected object, filtering the "measured" data by `Kalman Filter` ([docs](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)). Then, with those data I have created conditional algorithm that was based on the position and speed of points between successive frames. That algorithm return priorities with messages:
    * __0__ - BREAK
    * __1__ - SLOW DOWN
    * __2__ - BE CAREFUL
    * __3__ - GO AHEAD

As a result the algorithm takes the lowest priority and associated message of single frame.
