from __future__ import absolute_import, division, print_function

"""

This is a modification of the classify_images.py
script in Tensorflow. The original script produces
string labels for input images (e.g. you input a picture
of a cat and the script returns the string "cat"); this
modification reads in a directory of images and 
generates a vector representation of the image using
the penultimate layer of neural network weights.

Usage: python transflearn.py "../image_dir/*.jpg"

"""

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

import os.path
import re
import cv2
import sys
import tarfile
import glob
import json
import psutil
from collections import defaultdict
import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
MODEL_PATH = "transfLearning/NNs/inceptionv4/"
# pylint: enable=line-too-long

def download_and_extract(url, dest_directory):
  """Download and extract model tar file."""
  # dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


class InceptionV4:

  def __init__(self):
    self.sess = tf.Session()
    self.layer_dims = 2048
    self.pb_path = 'transfLearning/NNs/inceptionv4/classify_image_graph_def.pb'
    if not os.path.exists(self.pb_path):
      download_and_extract(DATA_URL, MODEL_PATH)
      # print(self.pb_path, "not exists")
      # raise ValueError

    self.create_graph()

  def create_graph(self):
    with self.sess.as_default():
      with tf.gfile.FastGFile(self.pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

  def run_inference_on_images(self, image_list, output_dir):
    """Runs inference on an image list.

    Args:
      image_list: a list of images.
      output_dir: the directory in which image vectors will be saved

    Returns:
      image_to_labels: a dictionary with image file keys and predicted
        text label values
    """
    image_to_labels = defaultdict(list)
    with self.sess:
      # Some useful tensors:
      # 'softmax:0': A tensor containing the normalized prediction across
      #   1000 labels.
      # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
      #   float description of the image.
      # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
      #   encoding of the image.
      # Runs the softmax tensor by feeding the image_data as input to the graph.
      softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')

      for image_index, image in enumerate(image_list):
        print("parsing", image_index, image)
        if not tf.gfile.Exists(image):
          tf.logging.fatal('File does not exist %s', image)
        
        with tf.gfile.FastGFile(image, 'rb') as f:
          image_data =  f.read()

          # predictions = sess.run(softmax_tensor,
          #                 {'DecodeJpeg/contents:0': image_data})

          # predictions = np.squeeze(predictions)

          ###
          # Get penultimate layer weights
          ###
          
          feature_tensor = self.sess.graph.get_tensor_by_name('pool_3:0')
          feature_set = self.sess.run(feature_tensor,
                          {'DecodeJpeg/contents:0': image_data})
          feature_vector = np.squeeze(feature_set)       
          outfile_name = os.path.basename(image) + ".npz"
          out_path = os.path.join(output_dir, outfile_name)
          np.savetxt(out_path, feature_vector, delimiter=',')

    return image_to_labels

  def run_inference_on_image(self, image):
    
    with self.sess.as_default():
      # Some useful tensors:
      # 'softmax:0': A tensor containing the normalized prediction across
      #   1000 labels.
      # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
      #   float description of the image.
      # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
      #   encoding of the image.
      # Runs the softmax tensor by feeding the image_data as input to the graph.
      image = cv2.resize(image, (299, 299))

      softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')

      # if not tf.gfile.Exists(image):
      #   tf.logging.fatal('File does not exist %s', image)
      #   return None
        
      # print("Get Ready!")

      feature_tensor = self.sess.graph.get_tensor_by_name('pool_3:0')
      # print(feature_tensor)
      feature_set = self.sess.run(feature_tensor,
              {'Sub:0': np.expand_dims(image, axis=0)})
      feature_vector = np.squeeze(feature_set)       
      # print(feature_vector) 
    return feature_vector

def main(_):
  
  if len(sys.argv) < 2:
    print("please provide a glob path to one or more images, e.g.")
    print("python classify_image_modified.py '../cats/*.jpg'")
    sys.exit()

  else:
    output_dir = "image_vectors"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    images = glob.glob(sys.argv[1])
    print(images)
    image_to_labels = run_inference_on_images(images, output_dir)
    print(output_dir)

    with open("image_to_labels.json", "w") as img_to_labels_out:
      json.dump(image_to_labels, img_to_labels_out)

    print("all done")
if __name__ == '__main__':
  tf.app.run()