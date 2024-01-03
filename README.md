# Dangerous-situations-with-pedastrians

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project is my thiesis project: "3D point cloud analysis for traffic situational awareness". Project mainly consist of predicting dangerous situations on the road with pedastrians. Data are provided by Ouster OS1 LiDAR. Needed files to run code: '.pcap', '.json'.

Result video is shown below:

<p align="center">
  <img src="results_mp4/wynik.gif" alt="animated" />
</p>
	
## Technologies
Project is created with:
* Python: 3.9
* numpy: 1.25.2
* opencv-python: 4.8.1.78
* ouster-sdk: 0.10.0
* roboflow: 1.1.9
* ultralytics: 8.0.208
	
## Setup
- **Clone** this repository to your local machine. Run this command inside your terminal:

    ```bash
    git clone https://github.com/SzewczykSzy/Dangerous-situations-with-pedastrians.git
    ```
- Open this repository localy.

### Usage
- To run the code write in terminal:
	```bash
 	python ./detect_dangerous_situations.py --weights weights/best_3000_s_100.pt --pcap-path ../data/PKR_test1/test4.pcap --metadata-path ../data/PKR_test1/test4.json --tracker ./trackers/bytetrack.yaml --imgsz 1024 --device cpu  --save=0 --save-video-path C:/Users/user/Ouster/Dangerous-situations-with-pedastrians/results_mp4/result.mp4    
 	```
