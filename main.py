"""Convert video files to tfrecords."""
import flow_vis
import os
import math
import random
import logging
import cv2
import numpy as np
import utils.flow_utils as FlowUtils
import utils.tfrecord_utils as TFRecordUtils
import utils.video_utils as VideoUtils
import utils.id_utils as IdUtils
import matplotlib.pyplot as plt

FILENAME_SUFFIX = '.tfrecord'
VIDEO_SUFFIX = '.avi'

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%m/%d/%Y %I:%M:%S %p')

def visualize_flows(frames, flows, num_imgs, video_save_path=None):
    
    combined = list(zip(frames, flows))
    enumerated = list(enumerate(combined))
    random_samples = random.choices(enumerated[1:len(enumerated)-1], k=num_imgs)
    
    f, axes = plt.subplots(math.ceil(num_imgs/4), 4, figsize = (60, 40))
    
    print("Raw Images")
    for num, axs in enumerate(axes):
        for i, ax in enumerate(axs):
            index = (num * 4 + i)
            if  index < num_imgs:
                img = random_samples[index][1][0]
                ax.imshow(img, aspect='auto')
                
    plt.show()
              
    print("FlowNet2 Images")
    f2, axes2 = plt.subplots(math.ceil(num_imgs/4), 4, figsize = (60, 40))
    for num, axs in enumerate(axes2):
        for i, ax in enumerate(axs):
            index = (num * 4 + i)
            if  index < num_imgs:
                flow_color = flow_vis.flow_to_color(random_samples[index][1][1], convert_to_bgr=False)
                ax.imshow(flow_color, aspect='auto')
                
    plt.show()
                
    print("OpenCV Images")
    f3, axes3 = plt.subplots(math.ceil(num_imgs/4), 4, figsize = (60, 40))
    for num, axs in enumerate(axes3):
        for i, ax in enumerate(axs):
            index = (num * 4 + i)
            if  index < num_imgs:
                img_index = random_samples[index][0]
                frame1 = frames[img_index]
                frame2 = frames[img_index + 1]
                flow = FlowUtils.calc_opt_flow(frame1, frame2)
                flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
                ax.imshow(flow_color, aspect='auto')

    plt.show()
    
    if video_save_path:
        VideoUtils.save_flow_video(flows, video_save_path)

def create_tfrecord(dataset, participant, day, hour, write_path, video_dir=None):
    # Read video
    timestamps, frames = dataset.get_frames(participant, day, hour)

    # Read labels
    labels = dataset.get_labels(participant, day, hour)

    logging.info("Computing optical flow")
    # Compute optical flow and remove first frame
    flows = FlowUtils.calc_approx_opt_flows(frames)
    frames = frames[1:]
    
    logging.info("Visualizing flows")
    id_str = IdUtils.get_id(participant, day, hour)
    video_save_path = os.path.join(video_dir, id_str + VIDEO_SUFFIX)
    visualize_flows(frames, flows, 20, video_save_path)

    logging.info("Writing .tfrecords to disk")
    # Write
    TFRecordUtils.write(participant, day, hour, timestamps, frames, flows, labels, write_path)

def main(dataset, labels_root, frames_root, export_dir, video_dir=None):
    """Main"""
    

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # Session ids
    data_info = dataset.get_data_info()
    logging.info("Loading {0} participant directories.".format(str(len(data_info))))

    # Create a separate TFRecords file for each video
    participants = data_info.keys()
    for participant in participants:
        for day in data_info[participant]:
            for hour in data_info[participant][day]:
                
                id_str = IdUtils.get_id(participant, day, hour)
                logging.info("Working on {}".format(id_str))

                # Output path
                write_path = os.path.join(export_dir, id_str + FILENAME_SUFFIX)

                # Check if file already generated
                if os.path.exists(write_path):
                    logging.info("Dataset file already exists. Skipping {0}.".format(id))
                    continue
                
                # Create record
                create_tfrecord(dataset, participant, day, hour, write_path, video_dir)

    # Print info
    dataset.done()
    logging.info("Finished converting the dataset!")