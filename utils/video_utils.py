import cv2
import os
import numpy as np
import flow_vis
import logging

def save_flow_video(flow_array, save_path):
    def get_img_size(image):
        height, width, _ = image.shape
        return width, height
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    flow_imgs = []
    for flow in flow_array:
        flow_img = flow_vis.flow_to_color(flow, convert_to_bgr=False)
        flow_imgs.append(flow_img)
        
    fps = 10
    size = get_img_size(flow_imgs[0])
    
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(flow_imgs)):
        out.write(flow_imgs[i])
        
        if i % 2500 == 0:
            logging.info('Building video: {0}'.format(i))
            
    out.release()
    
    print("Video saved at " + save_path)