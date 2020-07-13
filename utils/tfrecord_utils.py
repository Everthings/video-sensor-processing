import tensorflow as tf
import logging

def _int64_feature(value):
    """Return int64 feature"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Return bytes feature"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    """Return float feature"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write(participant, day, hour, timestamps, frames, flows, labels, path):    
    # Assertions
    if flows is not None:
        assert (frames.shape[0]==flows.shape[0]), \
        "Frame and optical flow length must match!"
        
    # Grabbing the dimensions
    num = frames.shape[0]
    
    video_id = "%s_%s_%s" % (participant, day, hour)
    
    # Writing
    counter = 0
    with tf.io.TFRecordWriter(path) as tfrecord_writer:
        for index in range(num):
            image_raw = frames[index].tostring()
            label = 1 if timestamps[index] in labels else 0
            feature = {
                'example/video_id': _bytes_feature(video_id.encode('utf-8')),
                'example/seq_no': _int64_feature(index),
                'example/label': _int64_feature(label),
                'example/image': _bytes_feature(image_raw)
            }
            if flows is not None:
                flow_arr = flows[index].ravel()
                feature['example/flow'] = _floats_feature(flow_arr)

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            tfrecord_writer.write(example.SerializeToString())
            
            counter += 1
            if counter % 2500 == 0:
                logging.info('Writing .tfrecords: {0}'.format(counter))
