import cv2
import numpy as np
import logging
import math
import torch
import torch.nn as nn
from collections import namedtuple

from utils.flownet2_pytorch.models import FlowNet2 

def calc_opt_flow(frame1, frame2):
    """Calculate optical flow as Dual TV-L1"""
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    dual_tv = cv2.optflow.createOptFlow_DualTVL1()
    return dual_tv.calc(frame1, frame2, None)

def calc_opt_flows(frames):
    """Calculate optical flow as Dual TV-L1"""
    dual_tv = cv2.optflow.createOptFlow_DualTVL1()
    frame_1 = frames[0]
    prvs = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
    flows = np.empty((frames.shape[0]-1, frames.shape[1], frames.shape[2], 2), np.dtype('float32'))
    i = 0
    
    for frame_2 in frames[1:]:
        nxt = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)
        flow = dual_tv.calc(prvs, nxt, None)
        flows[i] = flow
        prvs = nxt
        i += 1
        
        if i % 3 == 0:
            logging.info('Computing optical flow: {0}'.format(i))
        
    return flows

def calc_approx_opt_flows(frames):    
    mock_args = namedtuple('mock_args', 'fp16 rgb_max')
    args = mock_args(False, 255.0)
    
    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    state_dict = torch.load("./utils/flownet2_pytorch/models/FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(state_dict["state_dict"])
    
    flows = np.empty((frames.shape[0]-1, frames.shape[1], frames.shape[2], 2), np.dtype('float32'))
    
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        net.cuda()
        
    # taken from https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    def batch(iterable, n=1): 
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
            
    def get_img_batch(indices):
        imgs = []
        for index in indices:
            img_pair = [frames[index - 1], frames[index]]
            img_pair = np.array(img_pair).transpose(3, 0, 1, 2)
            imgs.append(img_pair)
        return np.array(imgs, np.dtype('float32'))
            
    batch_size = 60
    counter = 0
    for i in batch(range(1, len(frames)), batch_size):
        img_batch = torch.from_numpy(get_img_batch(i)).cuda()
        results = net(img_batch)
        
        for j in range(len(results)):
            result = results[j]
            flow = result.data.cpu().numpy().transpose(1, 2, 0)
            flows[counter] = flow
            counter += 1
        
            if counter % 2500 == 0:
                logging.info('Computing optical flow: {0}'.format(counter))
        
    return flows