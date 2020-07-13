"""Calorie-Harmony dataset"""
import os
import logging
import pandas as pd
import cv2
import numpy as np

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
  datefmt='%H:%M:%S', level=logging.INFO)

TRAIN_SPLIT = {'P429': {'2020_03_03': ['07', '08', '09'], '2020_03_04': ['05', '10', '12'], '2020_03_05': ['06', '11']}}
TEST_SPLIT = {'P431': {'2020_03_12': ['09', '10', '11', '12']}}

class Dataset():
    def __init__(self, labels_root_dir, frames_root_dir):
        self.labels_root_dir = labels_root_dir
        self.frames_root_dir = frames_root_dir

    def get_data_info(self):
        all_info = {**TRAIN_SPLIT, **TEST_SPLIT}
        return all_info

    def get_labels(self, participant, day, hour):
        """Read labels from csv to numpy array"""       
        labels_path = os.path.join(self.labels_root_dir, 'Labels/%s/%s/%s/Andy/gesture_labels.csv' % (participant, day, hour))
        labels = pd.read_csv(labels_path)
        labels = labels.dropna()
        labels = labels.loc[labels['certainty'] == 1.0]
        labels_timestamp = labels['timestamp'].values.tolist()
        return labels_timestamp
    
    def get_frames(self, participant, day, hour):
        """Convert a video file to numpy array"""
        
        def read_image(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            dim = (320, 256)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            
            return img
        
        frames_path = os.path.join(self.frames_root_dir, 'CalorieHarmony/%s/In_Wild/Camera/Frame/%s/%s' % (participant, day, hour))
        paths = os.listdir(frames_path)
        paths.sort()
        
        assert len(paths) != 0, "Couldn't find frames in directory"
        
        timestamps = []
        frames = []
        counter = 0
        for file in paths:
            if file.endswith(".jpg"):
                file_path = os.path.join(frames_path, file)
                timestamps.append(int(file.replace('.jpg', '')))
                frames.append(read_image(file_path))
                
                counter += 1
                if counter % 5000 == 0:
                    logging.info('Loading images: {0}'.format(counter))

        return timestamps, np.array(frames)

    def done(self):
        logging.info("Done")

    def get_train_split(self):
        return TRAIN_SPLIT

    def get_test_split(self):
        return TEST_SPLIT
