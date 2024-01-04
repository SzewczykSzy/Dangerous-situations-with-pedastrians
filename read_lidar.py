import cv2
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ouster import client
from ouster import pcap
from ouster.client import Scans, XYZLut, SensorInfo, destagger
from ouster.client.data import LidarScan
from ultralytics import YOLO
from contextlib import closing
from collections import defaultdict
from functions.filters import kalman_filter
from functions.equations import danger_sit


class YOLOModel:
    """
    Class for initialization of YOLO model and results of tracking.
    """
    def __init__(self, weights_path:str='./weights/best_3000_s_100.pt', imgsz=1024, tracker_path='trackers/bytetrack.yaml', persist=True, verbose=False):
        """
        Initialization

        Args:
            weights_path (str, optional): Path to weights file. Defaults to './weights/best_3000_s_100.pt'.
            imgsz (int, optional): Image size. Defaults to 1024.
            tracker_path (str, optional): Path to tracker file. Defaults to 'trackers/bytetrack.yaml'.
            persist (bool, optional): Defaults to True.
            verbose (bool, optional): If print result of track to terminal. Defaults to False.
        """
        self.model = YOLO(weights_path)
        self.persist = persist
        self.imgsz = imgsz
        self.tracker = tracker_path
        self.verbose = verbose

    def track(self, source:np.ndarray)-> tuple:
        """
        Tracking objects.

        Args:
            source (np.ndarray): Image on witch tracker works.

        Returns:
            (tuple): List including boxes (xyxy) of tracked objects and list of their ids.
        """
        results = self.model.track(source=source, persist=self.persist, imgsz=self.imgsz, tracker=self.tracker, verbose=self.verbose)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        if (results[0].boxes.id == None):
            ids = ''
        else:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
        return boxes, ids


class DataHandler:
    """
    Class for some data.
    """
    def __init__(self, metadata_path, pcap_path):
        """
        Initialization.

        Args:
            metadata_path (str): Path to `.json` file.
            pcap_path (str): Path to `.pcap` file.
        """
        self.metadata = SensorInfo(open(metadata_path, 'r').read())
        self.pcap_file = pcap.Pcap(pcap_path, self.metadata)
        self.xyz_lut = client.XYZLut(self.metadata)
        self.fps = int(str(self.metadata.mode)[-2:])
        self.width = int(str(self.metadata.mode)[:4])
        self.height = int(str(self.metadata.prod_line)[5:])
        self.output_dict = {3:["GO AHEAD", (0, 255, 0)], 2:["BE CAREFUL", (0, 155, 100)], 1:["SLOW DOWN", (0, 100, 155)], 
                   0:["BREAK", (0, 0, 255)]}
    
    def get_metadata(self):
        """
        Return `SensorInfo` object.

        Returns:
            (SensorInfo): Metadata.
        """
        return self.metadata

    def get_scans(self):
        """
        Return `Scans` object.

        Returns:
            (Scans): Scans.
        """
        scans = Scans(self.pcap_file)
        return scans

    def get_output_dict(self):
        """
        Return output dictionary (priority and associated message).

        Returns:
            (dict): output_dict.
        """
        return self.output_dict

    def get_xyz_lut(self):
        """
        Return xyz lookup table for transforming to cartesian coordinate system.

        Returns:
            (XYZLut): xyz_lut.
        """
        return self.xyz_lut
    
    def get_video_params(self):
        """
        Return parameters required for saving a video.

        Returns:
            (tuple): fps, image width, image height
        """
        return self.fps, self.width, self.height


class HistoryTracker:
    """
    Class for remembering track history.
    """
    def __init__(self):
        """
        Initialization.
        """
        self.history = defaultdict(lambda: [])

    def update_history(self, id, xyz_val):
        """
        Appending coordinates of object to track.

        Args:
            id (int): _description_
            xyz_val (list): _description_
        """
        track = self.history[id]
        if len(track) > 0:
            if xyz_val[0] == 0 or xyz_val[1] == 0:
                track.append(track[-1])
            else:
                track.append(xyz_val)
        else:
            track.append(xyz_val)
    
    def get_history(self):
        """Return track history.

        Returns:
            (defaultdict): dict with keys: `id`, values: lists of points.
        """
        return self.history

    def get_track(self, id):
        """Return object track history.

        Args:
            id (int): object's id

        Returns:
            (list): list of points
        """
        return self.history[id]


class HistoryTrackerXY:
    """Class for filtering track
    """
    def __init__(self):
        """_summary_
        """
        self.x = {}
        self.y = {}

    def update(self, track, id):
        """Updating KalamanFilter object, predicting next point. 

        Args:
            track (list): track of object
            id (int): object's id
        """
        value = self.x.get(id)
        if value is None:
            x_0 = track[0][0]
            y_0 = track[0][1]
            vx_0 = (track[1][0] - x_0)/0.1
            vy_0 = (track[1][1] - y_0)/0.1
            
            self.x[id] = kalman_filter(init=x_0, v_init=vx_0)
            self.y[id] = kalman_filter(init=y_0, v_init=vy_0)
            x = self.x[id]
            y = self.y[id]
        else:
            x = self.x[id]
            y = self.y[id]
        x.predict()
        y.predict()
        x.update([track[-1][0]])
        y.update([track[-1][1]])

    def get_x_data(self, id):
        """Return x coordinate value and velocity after filtering

        Args:
            id (int): object's id

        Returns:
            (KelmanFilter.x): predicted values related to x coordinate
        """
        return self.x[id].x
    
    def get_y_data(self, id):
        """Return y coordinate value and velocity after filtering

        Args:
            id (int): object's id

        Returns:
            (KelmanFilter.x): predicted values related to y coordinate
        """
        return self.y[id].x


class SingleFrame:
    """Class for representing single frame of data (scan).
    """
    def __init__(self, scan):
        """_summary_

        Args:
            scan (LidarScan): scan of environment
        """
        self.sig_field = scan.field(client.ChanField.SIGNAL)

    def get_combined_img(self, metadata):
        """Return combined image of black & white image. It is used to determine the legal dimensions of the detector input.

        Args:
            metadata (SensorInfo): metadata file

        Returns:
            (np.ndarray): 3D black & white image
        """
        sig_destaggered = destagger(metadata, self.sig_field)
        scaling_factor = 0.004
        constant = 0.5
        scaled_arr = sig_destaggered / (constant + scaling_factor * sig_destaggered)
        signal_image = scaled_arr.astype(np.uint8)
        combined_img = np.dstack((signal_image, signal_image, signal_image))
        return combined_img

    def get_xyz_destaggered(self, metadata, xyz_lut, scan):
        """Return destaggered scan

        Args:
            metadata (SensorInfo): metadata file
            xyz_lut (XYZLut): lookup table for transformation to cartesian coordinate system
            scan (LidarScan): single scan of environment

        Returns:
            (np.ndarray): A destaggered numpy array
        """
        xyz_destaggered = client.destagger(metadata, xyz_lut(scan))
        return xyz_destaggered


class VideoProcessor:
    """Class for processing video.
    """
    def __init__(self, metadata:SensorInfo):
        """_summary_

        Args:
            metadata (SensorInfo): metadata file
        """
        self.metadata = metadata

    def process_video(self, scan:LidarScan, model:YOLOModel, track_history:HistoryTracker, kalman:HistoryTrackerXY, xyz_lut:XYZLut, output_dict:dict, ):
        """Function processing single scan of data.

        Args:
            scan (LidarScan): single scan
            model (YOLOModel): detector
            track_history (HistoryTracker): track history of all objects 
            kalman (HistoryTrackerXY): filtered track 
            xyz_lut (XYZLut): lookup table
            output_dict (dict): contain priorities and the associated message

        Returns:
            (np.ndarray): image with annotations
        """
        frame = SingleFrame(scan)
        combined_img = frame.get_combined_img(self.metadata)
        xyz_destaggered = frame.get_xyz_destaggered(self.metadata, xyz_lut, scan)
        boxes, ids = model.track(source=combined_img)
        priority = 3
        distance = 0
        for box, id in zip(boxes, ids):
            center_x, center_y = int((box[0] + box[2])/2), int((box[1] + box[3])/2)
            xyz_val = xyz_destaggered[(center_y, center_x)]
            track_history.update_history(id=id, xyz_val=xyz_val)
            track = track_history.get_track(id)
            if len(track) >= 2:
                kalman.update(track, id)

                out, distance = danger_sit(kalman.get_x_data(id), kalman.get_y_data(id), id)

                if out < priority:
                    priority = out
            cv2.rectangle(combined_img, (box[0], box[1]), (box[2], box[3]), output_dict[priority][1], 2)
            cv2.rectangle(combined_img, (box[0], box[1]+2), (box[0]+160, box[1]-12), (255, 255, 255), -1)
            cv2.putText(combined_img, f"Id {id}; dist: {distance:0.2f} m", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(combined_img, f"{output_dict[priority][0]}", (470, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, output_dict[priority][1], 2)
        return combined_img
