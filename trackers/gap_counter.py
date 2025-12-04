from .base_tracker import TrackingAlgorithm
import config

class GapCounterTracker(TrackingAlgorithm):
    def __init__(self, args):
        super().__init__(args)
        self.gap_seconds = getattr(args, 'gap', config.GAP_SECONDS)
        # Store last seen frame per class: {cls_name: frame_idx}
        self.last_seen_frame = {}
        # We override standard event_counts logic for this specific algo
        self.gap_event_counts = {} 

    def get_gap_event_id(self, cls_name, current_frame, fps):
        gap_frames = int(self.gap_seconds * fps)
        
        if cls_name not in self.gap_event_counts:
            self.gap_event_counts[cls_name] = 1
            self.last_seen_frame[cls_name] = current_frame
            return 1

        last_frame = self.last_seen_frame[cls_name]
        delta = current_frame - last_frame
        self.last_seen_frame[cls_name] = current_frame # Update last seen

        if delta > gap_frames:
            self.gap_event_counts[cls_name] += 1
            
        return self.gap_event_counts[cls_name]

    def process_frame(self, frame, frame_idx):
        # This tracker does NOT track movement (x,y), it tracks "Events"
        results = self.model.predict(frame, classes=self.classes, verbose=False, conf=self.conf)
        
        processed_dets = []
        
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Logic: We use the FPS from the video cap inside the runner
                # But here we need to approximate or pass it down. 
                # For simplicity, we assume 30 or fetch from self (if we stored it).
                fps = 30 # Ideally passed from parent
                
                # Use class name as key for Gap Logic
                event_id = self.get_gap_event_id(cls_name, frame_idx, fps)
                
                # We return event_id as the "raw_id" so the parent class draws it as ID
                processed_dets.append((x1, y1, x2, y2, event_id, conf, cls_id))
        
        return processed_dets