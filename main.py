import argparse
import config
from trackers.yolo_native import YoloNativeTracker
from trackers.deep_sort import DeepSortTracker
from trackers.strong_sort import StrongSortTracker
from trackers.gap_counter import GapCounterTracker

def get_tracker_class(algo_name):
    """Factory function to return the correct class."""
    if algo_name in ['botsort', 'bytetrack']:
        return YoloNativeTracker
    elif algo_name == 'deepsort':
        return DeepSortTracker
    elif algo_name == 'strongsort':
        return StrongSortTracker
    elif algo_name == 'gap':
        return GapCounterTracker
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def main():
    parser = argparse.ArgumentParser(description="Unified Object Tracking Framework")
    
    # Core Arguments
    parser.add_argument('--algo', type=str, required=True, 
                        choices=['botsort', 'bytetrack', 'deepsort', 'strongsort', 'gap'],
                        help='Tracking Algorithm to use')
    parser.add_argument('--source', type=str, required=True, help='Path to video or 0 for webcam')
    
    # Optional settings
    parser.add_argument('--model', type=str, default=config.DEFAULT_MODEL, help='Path to .pt model')
    parser.add_argument('--conf', type=float, default=config.CONF_THRESHOLD, help='Confidence threshold')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class ID (e.g. 0 2)')
    parser.add_argument('--nosave', action='store_true', help='Do not save output video')
    parser.add_argument('--show', action='store_true', help='Show processing window')
    parser.add_argument('--draw-trails', action='store_true', help='Draw trajectories behind objects')
    
    # Gap specific
    parser.add_argument('--gap', type=float, default=config.GAP_SECONDS, help='Time gap for gap tracker')

    args = parser.parse_args()

    # Get the class and instantiate it
    try:
        TrackerClass = get_tracker_class(args.algo)
        tracker_instance = TrackerClass(args)
        tracker_instance.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()