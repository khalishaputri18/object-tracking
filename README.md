# Object and Frame Tracking
## Description
This project consists of:
1. **Multi-Object Trackers (Bytetrack, Botsort, Deepsort, Strongsort)**
   
   **Goal:** Track every individual object separately

   **Behavior:** If there are 5 people in the frame, the code assigns 5 unique IDs (e.g., Person 1, Person 2, Person 3...).
   
3. **Frame Tracker (Gap)**
   
   **Goal:** Track "Events" or "Appearances" of a class, not individual objects.
   
   **Behavior:** If there are 5 people in the frame, they all get the same ID (e.g., "Event 1"). The ID only increases if the object disappears for a specific time `--gap` and then reappears (starting "Event 2").
   

## How to use this project
### 1. Installation
```
pip install -r requirements.txt
```
### 2. Arguments
Core arguments details:
1. `--algo` = (**str**) Tracking algorithm to use ('botsort', 'bytetrack', 'deepsort', 'strongsort', or 'gap')
2. `--source` = (**str**) Path to video or 0 for webcam

Optional argument details:
1. `--model` = (**str**) Path to .pt model
2. `--conf` = (**float**) Confidence threshold
3. `--classes` = (**int**) Filter by class ID (e.g. 0 1)
4. `--nosave` = Do not save output video
5. `--show` = Show processing window
6. `--draw-trails` = Draw trajectory

Gap specific argument:
1. `--gap` = (**float**) Time gap (seconds)

### 3. How to run
Command example for object tracker:
```
python main.py --algo bytetrack --source my_video.mp4 --conf 0.5 --classes 0 1 2
```

Command example for frame tracker:
```
python main.py --algo gap --source my_video.mp4 --gap 5.0
