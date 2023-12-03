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


def main_loop(scans:Scans, xyz_lut:XYZLut, metadata:SensorInfo, colors_dict:dict, output_dict:dict, model:YOLO) -> None:
    for scan in scans:
        # Create signal image
        sig_field = scan.field(client.ChanField.SIGNAL)
        sig_destaggered = destagger(metadata, sig_field)
    
        # Define a scaling factor based on the values (adjust this as needed)
        scaling_factor = 0.004
        scaled_arr = sig_destaggered / (0.5 + scaling_factor * sig_destaggered) # BASIC

        # Convert to uint8 and create 3 dim matrix
        signal_image = scaled_arr.astype(np.uint8)
        combined_img = np.dstack((signal_image, signal_image, signal_image))

        # Create range image (for localization, distance measurement)
        range_field = scan.field(client.ChanField.RANGE)
        range_image = client.destagger(metadata, range_field)

        # xyz_destaggered = xyzlut(range_field)
        xyz_destaggered = client.destagger(metadata, xyz_lut(scan)) #to adjust for the pixel staggering that is inherent to Ouster lidar sensor raw data

        # Predict and track
        results = model.track(source=combined_img, persist=True, imgsz=1024, tracker='bytetrack.yaml')

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        if (results[0].boxes.id == None):
            ids = ''
        else:
            ids = results[0].boxes.id.cpu().numpy().astype(int)

        OUTPUT = 0
        color = (0, 255, 0)
        distance = 0

        # iteration through identified objects
        for box, id in zip(boxes, ids):

            center_x, center_y = int((box[0] + box[2])/2), int((box[1] + box[3])/2) #

            xyz_val = xyz_destaggered[(center_y, center_x)] #get the (x,y,z) coordinates with the lookup table 

            track = track_history[id] #save the (x,y,z) coordinates for distance calculation

            # Czasem zwraca wartość 0, jeśli tak, to podstawiam poprzednią wartość
            if len(track) > 0:
                if abs(xyz_val[0] - track[-1][0]) > 4 or abs(xyz_val[1] - track[-1][1]) > 4:
                    track.append(track[-1])
                else:
                    track.append(xyz_val)
            else:
                track.append(xyz_val)

            # ------------------------------------------------KALMAN--------------------------------------------------------------------------
            # Sprawdź, czy id w słowniku już istnieje, jeśli nie, to tworzy obiekti KalmanFilter z wartościami początkowymi
            if len(track) >= 2:
                
                x_0 = track[0][0]
                y_0 = track[0][1]
                vx_0 = (track[1][0] - x_0)/0.1
                vy_0 = (track[1][1] - y_0)/0.1
                value = track_history_filtered_x.get(id)

                kalman_output = kalman_history[id]

                # initialization of KalmanFilter object
                if value is None:
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

                out_x = track_filtered_x.x
                out_y = track_filtered_y.x

                number_of_detections += 1
                sum_of_accelerations += (track_filtered_x.x_post[1][0] - track_filtered_x.x_prior[1][0])/dt

                kalman_output.append([out_x[0][0], out_x[1][0], idx])

                out, distance = danger_sit(out_x, out_y, id)
            # ------------------------------------------------KALMAN--------------------------------------------------------------------------
                if out > OUTPUT:
                    OUTPUT = out

            cv2.rectangle(combined_img, (box[0], box[1]), (box[2], box[3]), COLORS_dict[OUTPUT], 2)
            cv2.rectangle(combined_img, (box[0], box[1]+2), (box[0]+160, box[1]-12), (255, 255, 255), -1)
            cv2.putText(combined_img, f"Id {id}; dist: {distance:0.2f} m", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(combined_img, f"{OUTPUT_dict[OUTPUT]}", (470, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_dict[OUTPUT], 2)

        # Stream results
        if True:
            cv2.imshow("YOLOv8 Tracking", combined_img)
            cv2.waitKey(1)  # 1 millisecond

        vid_writer.write(combined_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    OUTPUT_dict = {0:"GO AHEAD", 1:"BE CAREFUL", 2:"SLOW DOWN", 3:"BREAK", 4:"MISTAkE"}
    COLORS_dict = {0:(0, 255, 0), 1:(0, 155, 100), 2: (0, 100, 155), 3:(0, 0, 255), 4:(255, 0, 0)}

    # Load the YOLOv8 model
    model = YOLO('weights/best_s.pt')

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
    track_history = defaultdict(lambda: [])
    track_history_filtered_x = {}
    track_history_filtered_y = {}
    x_speed_history = []

    with closing(Scans(pcap_file)) as scans:

        save_path = "C:/Users/szyme/Ouster/YOLOv8/results_mp4"
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        main_loop(scans=scans, xyz_lut=xyz_lut, metadata=metadata, colors_dict=COLORS_dict,output_dict=OUTPUT_dict, model=model)

        vid_writer.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()