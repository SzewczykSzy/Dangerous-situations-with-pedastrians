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

class SingleFrame():
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

    def get_xyz_destaggered(self, metadata):
        xyz_destaggered = client.destagger(metadata, xyz_lut(scan))
        return xyz_destaggered

class TrackXY():
    def __init__(self):
        self.x = {}
        self.y = {}

    def update(self, track):
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


class TrackHistory():
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


class ReadPcap():
    def __init__(self, weights:str, metadata_path:str, pcap_path:str):
        self.output_dict = {3:["GO AHEAD", (0, 255, 0)], 
                            2:["BE CAREFUL", (0, 155, 100)],
                            1:["SLOW DOWN", (0, 100, 155)],
                            0:["BREAK", (0, 0, 255)]}
        self.model = YOLO(weights)
        self.metadata_path = metadata_path
        self.pcap_path = pcap_path
        self.metadata = client.SensorInfo(open(self.metadata_path, 'r').read())
        self.pcap_file = pcap.Pcap(self.pcap_path, self.metadata)
        self.xyz_lut = client.XYZLut(self.metadata)
        self.track_history = TrackHistory()
        self.filtered_track = TrackXY()
