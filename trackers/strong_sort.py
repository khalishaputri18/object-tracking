import sys
import numpy as np
from pathlib import Path
from .base_tracker import TrackingAlgorithm
import config

try:
    from boxmot.trackers.strongsort.strongsort import StrongSort
except ImportError:
    print("Error: Boxmot not installed. Cannot run StrongSort.")
    sys.exit(1)

class StrongSortTracker(TrackingAlgorithm):
    def __init__(self, args):
        super().__init__(args)
        self.tracker = StrongSort(
            reid_weights=config.STRONGSORT_WEIGHTS,
            device='cuda:0', # Or 'cpu'
            half=False,
            max_age=30,
            n_init=5,
            min_conf=self.conf
        )

    def process_frame(self, frame, frame_idx):
        # 1. Detect
        results = self.model.predict(frame, conf=self.conf, classes=self.classes, verbose=False)
        
        dets_to_track = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls = int(box.cls)
                dets_to_track.append([x1, y1, x2, y2, conf, cls])
        
        dets_to_track = np.array(dets_to_track)
        if len(dets_to_track) == 0: dets_to_track = np.empty((0, 6))

        # 2. Update StrongSort
        # Output format: [x1, y1, x2, y2, id, conf, cls, index]
        tracked_objects = self.tracker.update(dets_to_track, frame)
        
        processed_dets = []
        for output in tracked_objects:
            x1, y1, x2, y2, raw_id, conf, cls, _ = output
            processed_dets.append((int(x1), int(y1), int(x2), int(y2), int(raw_id), float(conf), int(cls)))
            
        return processed_dets