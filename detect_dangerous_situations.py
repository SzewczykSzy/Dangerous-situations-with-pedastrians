"""
Run dangerous situation detection on pcap file

Usage:
    TODO:
                                                             
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
    parser.add_argument("--save-video-path", type=str, default='', help="Path for saving result video, i.e 'C:/Users/account/res.mp4'")
    parser.add_argument("--imgsz", action="store_true", help="View detected images")
    args = parser.parse_args()

    print("Class Index:", args.class_)
    print("Weights Path:", args.weights)
    print("Confidence Threshold:", args.conf_thres)
    print("Source PCAP Path:", args.source) 
    print("Metadata Path:", args.metadata_path)
    print("View Image:", args.view_img)
    return args

def run(save_video_path,
        weights='weights/yolov5s.pt',
        pcap_path='data/pcap/example.pcap',
        metadata_path='data/json/example.json',
        tracker = 'weights/bytetracker.yaml',
        imgsz = 1024,
        device = 'cpu',
        if_save = False,
        save_video_path = ''
        ):

    yolo_model = YOLOModel(weights)
    data_handler = DataHandler(metadata_path=metadata_path, pcap_path=pcap_path)
    metadata = data_handler.get_metadata()
    output_dict = data_handler.get_output_dict()
    video_params = data_handler.get_video_params()

    video_processor = VideoProcessor(metadata=metadata, save=if_save, save_path=save_video_path, video_params=video_params)
    history = HistoryTracker()
    kalman = HistoryTrackerXY()

    scans = data_handler.get_scans()
    xyz_lut = data_handler.get_xyz_lut()
    for scan in scans:
        video_processor.process_video(scan=scan, model=yolo_model, track_history=history, kalman=kalman, xyz_lut=xyz_lut, output_dict=output_dict)


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)