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


def main_loop(scans:Scans, xyz_lut:XYZLut, metadata:SensorInfo, output_dict:dict, 
              model:YOLO, track_history:defaultdict, track_history_filtered_x:dict, 
              track_history_filtered_y:dict, vid_writer:cv2.VideoWriter) -> None:
    
    # iteration through all scans
    for scan in scans:
        # Create signal image
        sig_field = scan.field(client.ChanField.SIGNAL)
        sig_destaggered = destagger(metadata, sig_field)
    
        # Define a scaling factor based on the values
        scaling_factor = 0.004
        scaled_arr = sig_destaggered / (0.5 + scaling_factor * sig_destaggered)

        # Convert to uint8 and create 3 dim matrix
        signal_image = scaled_arr.astype(np.uint8)
        combined_img = np.dstack((signal_image, signal_image, signal_image))

        # To adjust for the pixel staggering that is inherent to Ouster lidar sensor raw data
        xyz_destaggered = client.destagger(metadata, xyz_lut(scan))

        # Predict and track
        results = model.track(source=combined_img, persist=True, imgsz=1024, tracker='bytetrack.yaml', verbose=False)

        # Get predict boxes
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        if (results[0].boxes.id == None):

            ids = ''
        else:
            ids = results[0].boxes.id.cpu().numpy().astype(int)

        priority = 3
        distance = 0

        # iteration through identified objects
        for box, id in zip(boxes, ids):

            center_x, center_y = int((box[0] + box[2])/2), int((box[1] + box[3])/2) #

            xyz_val = xyz_destaggered[(center_y, center_x)] #get the (x,y,z) coordinates with the lookup table 

            track = track_history[id] #save the (x,y,z) coordinates for distance calculation

            # First small filter, error with returning 0
            if len(track) > 0:
                if xyz_val[0] == 0 or xyz_val[1] == 0:
                    track.append(track[-1])
                else:
                    track.append(xyz_val)
            else:
                track.append(xyz_val)

            # ------------------------------------------------KALMAN------------------------------------------
            if len(track) >= 2:
                
                value = track_history_filtered_x.get(id)

                # initialization of KalmanFilter object
                if value is None:
                    x_0 = track[0][0]
                    y_0 = track[0][1]
                    vx_0 = (track[1][0] - x_0)/0.1
                    vy_0 = (track[1][1] - y_0)/0.1
                    
                    track_history_filtered_x[id] = kalman_filter(init=x_0, v_init=vx_0)
                    track_history_filtered_y[id] = kalman_filter(init=y_0, v_init=vy_0)
                    track_filtered_x = track_history_filtered_x[id]
                    track_filtered_y = track_history_filtered_y[id]
                else:
                    track_filtered_x = track_history_filtered_x[id]
                    track_filtered_y = track_history_filtered_y[id]

                track_filtered_x.predict()
                track_filtered_y.predict()

                track_filtered_x.update([track[-1][0]])
                track_filtered_y.update([track[-1][1]])

                out, distance = danger_sit(track_filtered_x, track_filtered_y, id)
            # ------------------------------------------------KALMAN--------------------------------------------------------------------------
                if out < priority:
                    priority = out

            cv2.rectangle(combined_img, (box[0], box[1]), (box[2], box[3]), output_dict[priority][1], 2)
            cv2.rectangle(combined_img, (box[0], box[1]+2), (box[0]+160, box[1]-12), (255, 255, 255), -1)
            cv2.putText(combined_img, f"Id {id}; dist: {distance:0.2f} m", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(combined_img, f"{output_dict[priority][0]}", (470, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, output_dict[priority][1], 2)

        # Stream results
        if True:
            cv2.imshow("YOLOv8 Tracking", combined_img)
            cv2.waitKey(1)  # 1 millisecond

        vid_writer.write(combined_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    OUTPUT_dict = {3:["GO AHEAD", (0, 255, 0)], 2:["BE CAREFUL", (0, 155, 100)], 1:["SLOW DOWN", (0, 100, 155)], 
                   0:["BREAK", (0, 0, 255)]}

    # Load the YOLOv8 model
    model = YOLO('weights/best_3000_s_100.pt')

    # Paths to pcap and json files
    metadata_path = "C:/Users/szyme/Ouster/data/PKR_test1/test4.json"
    pcap_path = "C:/Users/szyme/Ouster/data/PKR_test1/test4.pcap"

    # Making PacketSource from data
    with open(metadata_path, 'r') as f:
        metadata = client.SensorInfo(f.read())

    fps = int(str(metadata.mode)[-2:])
    width = int(str(metadata.mode)[:4])
    height = int(str(metadata.prod_line)[5:])

    pcap_file = pcap.Pcap(pcap_path, metadata)

    xyz_lut = client.XYZLut(metadata) #call cartesian lookup table

    # Store the track history
    track_history = defaultdict(lambda: [])     # dictionary: {key=id, value=[xyz_val_0, ... ,xyz_val_99]}
    
    track_history_filtered_x = {}   # dictionary: {key=id, value=kalman_filter(init=0, v_init=0)}
    track_history_filtered_y = {}   # dictionary: {key=id, value=kalman_filter(init=0, v_init=0)}

    with closing(Scans(pcap_file)) as scans:

        save_path = "C:/Users/szyme/Ouster/Dangerous-situations-with-pedastrians/results_mp4/wynik.mp4"
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        main_loop(scans=scans, xyz_lut=xyz_lut, metadata=metadata, output_dict=OUTPUT_dict, 
                  model=model, track_history=track_history, track_history_filtered_x=track_history_filtered_x, 
                  track_history_filtered_y=track_history_filtered_y, vid_writer=vid_writer)

        vid_writer.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()