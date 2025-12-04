import os
import cv2
import datetime
import numpy as np
from pathlib import Path

def get_unique_filepath(directory, base_name, extension):
    """Generates a unique filename to avoid overwriting."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    counter = 0
    while True:
        suffix = f"_{counter}" if counter > 0 else ""
        filename = f"{base_name}{suffix}{extension}"
        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            return file_path
        counter += 1

def format_timestamp(seconds):
    """Track timestamp of the video where objects are tracked"""
    return str(datetime.timedelta(seconds=int(seconds)))

def generate_colors(seed=42, num_colors=1000):
    """Assign different color to every object ID"""
    np.random.seed(seed)
    return np.random.randint(0, 255, size=(num_colors, 3), dtype='uint8')