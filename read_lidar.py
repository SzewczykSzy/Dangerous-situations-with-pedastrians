'''
'''
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
    def __init__(self, weights_path:str='./weights/best_3000_s_100.pt', imgsz=1024, tracker_path='trackers/bytetrack.yaml', persist=True, verbose=False):
        self.model = YOLO(weights_path)
        self.persist = persist
        self.imgsz = imgsz
        self.tracker = tracker_path
        self.verbose = verbose

    def track(self, source:np.ndarray)-> tuple:
        results = self.model.track(source=source, persist=self.persist, imgsz=self.imgsz, tracker=self.tracker, verbose=self.verbose)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        if (results[0].boxes.id == None):
            ids = ''
        else:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
        return boxes, ids

class DataHandler:
    def __init__(self, metadata_path, pcap_path):
        self.metadata = SensorInfo(open(metadata_path, 'r').read())
        self.pcap_file = pcap.Pcap(pcap_path, self.metadata)
        self.xyz_lut = client.XYZLut(self.metadata)
        self.fps = int(str(self.metadata.mode)[-2:])
        self.width = int(str(self.metadata.mode)[:4])
        self.height = int(str(self.metadata.prod_line)[5:])
        self.output_dict = {3:["GO AHEAD", (0, 255, 0)], 2:["BE CAREFUL", (0, 155, 100)], 1:["SLOW DOWN", (0, 100, 155)], 
                   0:["BREAK", (0, 0, 255)]}
    
    def get_metadata(self):
        return self.metadata

    def get_scans(self):
        scans = Scans(self.pcap_file)
        return scans

    def get_output_dict(self):
        return self.output_dict

    def get_xyz_lut(self):
        return self.xyz_lut
    
    def get_video_params(self):
        return self.fps, self.width, self.height

class HistoryTracker:
    def __init__(self):
        self.history = defaultdict(lambda: [])

    def update_history(self, id, xyz_val):
        track = self.history[id]
        if len(track) > 0:
            if xyz_val[0] == 0 or xyz_val[1] == 0:
                track.append(track[-1])
            else:
                track.append(xyz_val)
        else:
            track.append(xyz_val)
    
    def get_history(self):
        return self.history

    def get_track(self, id):
        return self.history[id]


class HistoryTrackerXY:
    def __init__(self):
        self.x = {}
        self.y = {}

    def update(self, track, id):
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
        return self.x[id].x
    
    def get_y_data(self, id):
        return self.y[id].x


class SingleFrame:
    def __init__(self, scan):
        self.sig_field = scan.field(client.ChanField.SIGNAL)

    def get_combined_img(self, metadata):
        sig_destaggered = destagger(metadata, self.sig_field)
        scaling_factor = 0.004
        constant = 0.5
        scaled_arr = sig_destaggered / (constant + scaling_factor * sig_destaggered)
        signal_image = scaled_arr.astype(np.uint8)
        combined_img = np.dstack((signal_image, signal_image, signal_image))
        return combined_img

    def get_xyz_destaggered(self, metadata, xyz_lut, scan):
        xyz_destaggered = client.destagger(metadata, xyz_lut(scan))
        return xyz_destaggered


class VideoProcessor:
    def __init__(self, metadata):
        self.metadata = metadata

    def process_video(self, scan:LidarScan, model:YOLOModel, track_history:HistoryTracker, kalman:HistoryTrackerXY, xyz_lut:XYZLut, output_dict:dict, ):
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
