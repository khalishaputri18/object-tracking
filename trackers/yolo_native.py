from .base_tracker import TrackingAlgorithm

class YoloNativeTracker(TrackingAlgorithm):
    def process_frame(self, frame, frame_idx):
        # Decide tracker config based on algo name
        tracker_file = "botsort.yaml" if self.args.algo == "botsort" else "bytetrack.yaml"
        
        # Run Tracking
        results = self.model.track(
            frame, 
            persist=True, 
            tracker=tracker_file,
            conf=self.conf,
            classes=self.classes,
            verbose=False
        )

        processed_dets = []
        
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Extract data
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                raw_id = int(box.id[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                processed_dets.append((x1, y1, x2, y2, raw_id, conf, cls_id))
        
        return processed_dets