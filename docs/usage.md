- To run the code on your custom data, clone github repository:
```
git clone https://github.com/SzewczykSzy/Dangerous-situations-with-pedastrians.git
```

- Open this repository localy.

- Install all required packages:
```bash
pip install -r requirements.txt
```

- To run the code write in terminal:
```bash
python ./detect_dangerous_situations.py --weights weights/best_3000_s_100.pt --pcap-path ../PATH_TO_PCAP_FILE/sample.pcap --metadata-path ../PATH_TO_JSON_FILE/sample.json --tracker ./trackers/bytetrack.yaml --imgsz 1024 --device cpu --save=0 --save-video-path C:/PATH_TO_REPOSITORY/Dangerous-situations-with-pedastrians/results_mp4/result.mp4
```

!!! Warning

    To stop the code while running press `q` on keyboard.

::: detect_dangerous_situations.run

