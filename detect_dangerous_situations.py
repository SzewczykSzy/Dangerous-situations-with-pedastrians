"""
Run dangerous situation detection on pcap file

Usage:
    TODO:
                                                             
"""
import sys
import os
import argparse


def parse_opt():
    parser = argparse.ArgumentParser(description="PCAP File Detector using YOLOv5")
    parser.add_argument("--class_", type=int, help="Class index to detect", required=True)
    parser.add_argument("--weights", type=str, help="Path to YOLOv5 weights file", required=True)
    parser.add_argument("--conf-thres", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--source", type=str, help="Path to the PCAP file", required=True)
    parser.add_argument("--metadata-path", type=str, help="Path to metadata JSON file", required=True)
    parser.add_argument("--view-img", action="store_true", help="View detected images")
    args = parser.parse_args()

    print("Class Index:", args.class_)
    print("Weights Path:", args.weights)
    print("Confidence Threshold:", args.conf_thres)
    print("Source PCAP Path:", args.source)
    print("Metadata Path:", args.metadata_path)
    print("View Image:", args.view_img)
    return args

def run(weights='weights/yolov5s.pt',
        pcap_path='data/pcap/example.pcap',
        metadata_path='data/json/example.json',
        imgsz = 1024,
        device = 'cpu',
        save_video = 1
        ):
    
    pass

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)