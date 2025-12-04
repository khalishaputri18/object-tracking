import cv2
import time
import numpy as np
import os
from collections import deque
from ultralytics import YOLO
import config
from utils import get_unique_filepath, format_timestamp, generate_colors

class TrackingAlgorithm:
    def __init__(self, args):
        self.args = args
        self.source = args.source
        self.model_path = args.model
        self.conf = args.conf
        self.classes = args.classes
        
        # --- ID MAPPING STATE ("Skipping ID" Logic) ---
        # This ensures that even if the tracker gives ID 100, 
        # we show ID 1 if it's the first person we see.
        self.class_id_mappings = {} # {cls_id: {raw_id: visual_id}}
        self.class_counters = {}    # {cls_id: next_available_id}
        
        self.trajectories = {}
        self.id_colors = generate_colors()
        self.names = {}
        
        self.load_model()

    def load_model(self):
        print(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.names = self.model.names

    def get_video_capture(self):
        cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
        if not cap.isOpened():
            raise ValueError(f"Could not open source: {self.source}")
        return cap

    def initialize_outputs(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30 if (fps == 0 or np.isnan(fps)) else int(fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        base_name = f"{self.args.algo}_{os.path.splitext(os.path.basename(str(self.source)))[0]}"
        
        csv_path = get_unique_filepath(config.OUTPUT_DIR, base_name, ".csv")
        vid_path = get_unique_filepath(config.OUTPUT_DIR, base_name, ".mp4") if not self.args.nosave else None

        with open(csv_path, 'w') as f:
            f.write("Frame,Timestamp,Class,ID,Conf,x1,y1,x2,y2\n")

        writer = None
        if vid_path:
            writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        return fps, total_frames, csv_path, writer

    def get_visual_id(self, raw_id, cls_id):
        """
        Handles the 'Skipping ID' logic.
        Maps raw tracker IDs (which can be huge or random) to clean, sequential IDs per class.
        """
        if cls_id not in self.class_counters:
            self.class_counters[cls_id] = 1
            self.class_id_mappings[cls_id] = {}
        
        if raw_id not in self.class_id_mappings[cls_id]:
            self.class_id_mappings[cls_id][raw_id] = self.class_counters[cls_id]
            self.class_counters[cls_id] += 1
            
        return self.class_id_mappings[cls_id][raw_id]

    def update_trajectories(self, visual_id, centroid):
        if visual_id not in self.trajectories:
            self.trajectories[visual_id] = deque(maxlen=config.TRAIL_LENGTH)
        self.trajectories[visual_id].append(centroid)

    def draw_and_save(self, frame, results_list, frame_idx, timestamp, csv_file):
        with open(csv_file, 'a') as f:
            for x1, y1, x2, y2, raw_id, conf, cls_id in results_list:
                # 1. CLEAN ID LOGIC APPLIED HERE
                visual_id = self.get_visual_id(raw_id, cls_id)
                
                cls_name = self.names.get(cls_id, str(cls_id))
                line = f"{frame_idx},{timestamp},{cls_name},{visual_id},{conf:.2f},{x1},{y1},{x2},{y2}\n"
                f.write(line)

                color_seed = visual_id + (cls_id * 50)
                color = [int(c) for c in self.id_colors[color_seed % len(self.id_colors)]]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{cls_name} {visual_id}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if self.args.draw_trails:
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    self.update_trajectories(visual_id, (cx, cy))
                    path = self.trajectories[visual_id]
                    for i in range(1, len(path)):
                        if path[i-1] is None or path[i] is None: continue
                        thickness = int(np.sqrt(64 / float(len(path) - i + 1)) * 2)
                        cv2.line(frame, path[i-1], path[i], color, thickness)

    def process_frame(self, frame, frame_idx):
        raise NotImplementedError("Child class must implement process_frame")

    def run(self):
        cap = self.get_video_capture()
        fps, total_frames, csv_path, writer = self.initialize_outputs(cap)
        
        frame_idx = 0
        start_time = time.time() # For FPS calculation
        print(f"Started processing {self.source} using {self.args.algo}...")

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            timestamp = format_timestamp(frame_idx / fps)
            
            detections = self.process_frame(frame, frame_idx)
            self.draw_and_save(frame, detections, frame_idx, timestamp, csv_path)

            # --- PROGRESS LOGGING ---
            if frame_idx % 20 == 0:
                end_time = time.time()
                elapsed = end_time - start_time
                # Avoid division by zero
                current_fps = 20 / elapsed if elapsed > 0 else 0
                
                # Print to Console
                print(f"[{self.args.algo}] Frame {frame_idx}/{total_frames} | Speed: {current_fps:.1f} FPS")
                
                # Reset timer
                start_time = time.time()
            # ------------------------

            # HUD on Video
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            if writer: writer.write(frame)
            if self.args.show:
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print(f"\nProcessing Complete. Data saved to:\n - CSV: {csv_path}")