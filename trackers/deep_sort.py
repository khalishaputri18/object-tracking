from deep_sort_realtime.deepsort_tracker import DeepSort
from .base_tracker import TrackingAlgorithm

class DeepSortTracker(TrackingAlgorithm):
    def __init__(self, args):
        super().__init__(args)
        # Initialize DeepSort specifically
        self.tracker = DeepSort(max_age=15, n_init=5, nms_max_overlap=1.0)

    def process_frame(self, frame, frame_idx):
        # 1. Detect using YOLO
        results = self.model.predict(frame, conf=self.conf, classes=self.classes, verbose=False)
        
        # 2. Format for DeepSort: [[left, top, w, h], conf, detection_class]
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append([[x1, y1, w, h], conf, cls])

        # 3. Update Tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        processed_dets = []
        for track in tracks:
            if not track.is_confirmed(): continue
            
            raw_id = track.track_id # DeepSort usually gives string IDs
            cls_id = int(track.det_class)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            conf = track.det_conf if track.det_conf else 0.0
            
            processed_dets.append((x1, y1, x2, y2, raw_id, conf, cls_id))
            
        return processed_dets