## Requirements
Install ultralytics
```
pip install ultralytics
```

Install deep-sort-realtime if you want to use deepsort
```
pip install deep-sort-realtime
```

## How to run

### Frame-track
Arguments configuration
1. `source` = [string] video source directory or file
2. `model` = [string] yolo model used (.pt)
3. `classes` = [int] class_id that want to be processed
4. `gap` = [float] time gap (seconds)
5. `conf` = [float] the confidence threshold of object detected
7. `nosave` = don't save the video output file, only label
8. `show` = show preview window

Example command for frame-track:
```
python frame-track.py --model yolov8n.pt --source ../test/ --nosave --classes 5 6 --conf 0.5 --gap 3.0 --nosave
```
The command above will track and count class_id 5 and 6 of all .mp4 videos inside test folder in batch processing. The output videos will not be saved and the preview window will not be shown.

### Bytetrack and Botsort
Arguments configuration
1. `source` = [string] video source directory or file
2. `model` = [string] yolo model used (.pt)
3. `track` = activate tracking
4. `count` = activate counting
5. `batch` = activate batch processing (no streaming)
6. `classes` = [int] class_id that want to be processed
7. `nosave` = don't save the video output file, only label

Program configuration
1. `CONF_THRESHOLD` = [float] the confidence threshold of object detected
2. `MIN_HISTORY` = [int] how many minimum frames the object must appear before start being tracked.

Example command for bytetrack:
```
python bytetrack.py --model yolov8n.pt --source ../test/ --track --count --batch --nosave --classes 0 1 2 3 4 5 6
```
Example command for botsort:
```
python botsort.py --model yolov8n.pt --source ../test/ --track --count --batch --nosave --classes 0 1 2 3 4 5 6
```
The commands above will track and count class_id 0 to 6 of all .mp4 videos inside test folder in batch processing. The output videos will not be saved.

### Deepsort
Arguments configuration
1. `source` = [string] video source directory or file
2. `model` = [string] yolo model used (.pt)
3. `track` = activate tracking
4. `count` = activate counting
5. `batch` = activate batch processing (no streaming)
6. `classes` = [int] class_id that want to be processed
7. `nosave` = don't save the video output file, only label
8. `conf` = [float]the confidence threshold of object detected

Example command for deepsort:
```
python deepsort.py --model yolov8n.pt --source ../test/ --track --count --batch --nosave --classes 0 1 2 3 4 5 6 --conf 0.5
```
### Strongsort
1. `source` = [string] video source directory or file
2. `model` = [string] yolo model used (.pt)
3. `count` = activate counting
4. `batch` = activate batch processing (no streaming)
5. `classes` = [int] class_id that want to be processed
6. `nosave` = don't save the video output file, only label
7. `conf` = [float]the confidence threshold of object detected

Example command for strongsort:
```
python strongsort.py --model yolov8n.pt --source ../test/ --count --batch --nosave --classes 0 1 2 3 4 5 6 --conf 0.5
```
