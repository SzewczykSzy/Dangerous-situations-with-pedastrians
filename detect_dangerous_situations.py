"""
Run dangerous situation detection on pcap file

Usage:
    'python ./detect_dangerous_situations.py --weights weights/best_3000_s_100.pt 
    --pcap-path ../data/PKR_test1/test4.pcap --metadata-path ../data/PKR_test1/test4.json 
    --tracker ./trackers/bytetrack.yaml --imgsz 1024 --device cpu  --save=0 
    --save-video-path C:/Users/user/Ouster/Dangerous-situations-with-pedastrians/results_mp4/result.mp4'                                                       
"""

import sys
import os
import argparse
from functions.read_pcap import ReadPcap, SingleFrame
from functions.equations import danger_sit
from read_lidar import *


def parse_opt():
    parser = argparse.ArgumentParser(description="Dangerous situation detector with YOLOv8")
    parser.add_argument("--weights", type=str, help="Path to YOLOv8 weights file", required=True)
    parser.add_argument("--pcap-path", type=str, help="Path to source PCAP file", required=True)
    parser.add_argument("--metadata-path", type=str, help="Path to metadata JSON file", required=True)
    parser.add_argument("--tracker", type=str, help="Path to tracker YAML", required=True)
    parser.add_argument("--imgsz", type=int, help="Size of single frame", required=True)
    parser.add_argument("--device", type=str, default='cpu', help="Cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--save", type=int, default=0, help="If want to save result video: 1, else 0")
    parser.add_argument("--save-video-path", type=str, default='', help="Path for saving result video, i.e 'C:/Users/account/res.mp4'")
    args = parser.parse_args()

    print("Weights Path:", args.weights)
    print("Pcap file Path:", args.pcap_path)
    print("Metadata file Path:", args.metadata_path) 
    print("Tracker Path:", args.tracker)
    print("Image Size:", args.imgsz)
    print("Device:", args.device)
    print("Save:", args.save)
    print("Video path to save:", args.save_video_path)
    return args

def run(weights='weights/yolov5s.pt',
        pcap_path='data/pcap/example.pcap',
        metadata_path='data/json/example.json',
        tracker = 'weights/bytetracker.yaml',
        imgsz = 1024,
        device = 'cpu',
        save = False,
        save_video_path = ''
        ):

    yolo_model = YOLOModel(weights, imgsz=imgsz)
    data_handler = DataHandler(metadata_path=metadata_path, pcap_path=pcap_path)
    metadata = data_handler.get_metadata()
    output_dict = data_handler.get_output_dict()
    fps, width, height = data_handler.get_video_params()

    video_processor = VideoProcessor(metadata=metadata)
    history = HistoryTracker()
    kalman = HistoryTrackerXY()

    scans = data_handler.get_scans()
    xyz_lut = data_handler.get_xyz_lut()
    if save == 1:
        vid_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for scan in scans:
        image = video_processor.process_video(scan=scan, model=yolo_model, track_history=history, kalman=kalman, xyz_lut=xyz_lut, output_dict=output_dict)
        
        cv2.imshow("YOLOv8 Tracking", image)
        cv2.waitKey(1)  # 1 millisecond
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if save == 1:
            vid_writer.write(image)

    if save == 1:    
        vid_writer.release()
    cv2.destroyAllWindows()


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)