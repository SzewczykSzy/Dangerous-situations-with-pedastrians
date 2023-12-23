import cv2
from ouster.client import Scans, XYZLut, SensorInfo, destagger
from ouster.client.data import LidarScan
from ultralytics import YOLO
from contextlib import closing
from collections import defaultdict
from functions.filters import kalman_filter
from functions.equations import danger_sit


class LidarProcessor:
    def __init__(self, model_path, metadata_path, pcap_path, save_path):
        self.model = YOLO(model_path)
        self.metadata = SensorInfo(open(metadata_path, 'r').read())
        self.pcap_file = pcap.Pcap(pcap_path, self.metadata)
        self.xyz_lut = client.XYZLut(self.metadata)
        self.fps = int(str(self.metadata.mode)[-2:])
        self.width = int(str(self.metadata.mode)[:4])
        self.height = int(str(self.metadata.prod_line)[5:])
        self.save_path = save_path

        self.track_history = defaultdict(lambda: [])
        self.track_history_filtered_x = {}
        self.track_history_filtered_y = {}

    def process_scans(self):
        with closing(Scans(self.pcap_file)) as scans:
            vid_writer = cv2.VideoWriter(
                self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

            for scan in scans:
                # processing logic goes here

                vid_writer.write(combined_img)

            vid_writer.release()
            cv2.destroyAllWindows()


class Visualizer:
    @staticmethod
    def display_images():
        # Add code to display images using cv2.imshow() and cv2.waitKey()


def main():
    processor = LidarProcessor("weights/best_3000_s_100.pt", "path_to_metadata.json", "path_to_pcap.pcap", "output_path.mp4")
    processor.process_scans()
    Visualizer.display_images()


if __name__ == "__main__":
    main()