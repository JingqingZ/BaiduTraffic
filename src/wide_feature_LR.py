#!/usr/bin/python
# coding=utf-8

import os
import math
from collections import defaultdict
import cPickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

gpu_id = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(session)

"""
    Logistic Regression with wide feature
"""

root_dir = "../pkl/"

"""
link_num is 1275, feature_dim is 22(coarse)
link_num is 1275, feature_dim is 129(fine)
link_num is 1275, feature_dim is 68(link_info)
feature dim is 6(time_feature)
input_dim = 22 + 129 + 68 + 6 = 225
"""
input_poi_type_feature_coarse_file = root_dir + "each_link_top5_poi_type_feature_coarse_beijing.pkl"
input_poi_type_feature_fine_file = root_dir + "each_link_top5_poi_type_feature_fine_beijing.pkl"
input_event_top5_link_info_feature_beijing_file = root_dir + "event_top5_link_info_feature_beijing.pkl"
input_time_feature_file = root_dir + "time_feature.pkl"
input_traffic_file = root_dir + "event_traffic_completion_beijing.pkl"

(link_list_coarse, poi_type_feature_coarse) = cPickle.load(open(input_poi_type_feature_coarse_file, "rb"))
(link_list_fine, poi_type_feature_fine) = cPickle.load(open(input_poi_type_feature_fine_file, "rb"))
(link_list, link_info_feature) = cPickle.load(open(input_event_top5_link_info_feature_beijing_file, "rb"))
time_feature = cPickle.load(open(input_time_feature_file))
event_traffic = cPickle.load(open(input_traffic_file))
TOTAL_TIME = 61 * 24 * 12
train_time = int(TOTAL_TIME * 0.7)
test_time = TOTAL_TIME - train_time
link_num = len(link_list)
input_dim = 225
coarse_feature_dim = np.shape(poi_type_feature_coarse)[1]
fine_feature_dim = np.shape(poi_type_feature_fine)[1]
link_info_feature_dim = np.shape(link_info_feature)[1]
time_feature_dim = np.shape(time_feature)[1]


# normalize the feature matrix, 2 dim
def data_normalize(feature_matrix):
    miu = np.mean(feature_matrix, axis=0)
    std = np.std(feature_matrix, axis=0)
    feature_matrix = (feature_matrix - miu) / std
    return feature_matrix


poi_type_feature_coarse = data_normalize(poi_type_feature_coarse)
poi_type_feature_fine = data_normalize(poi_type_feature_fine)
link_info_feature = data_normalize(link_info_feature)
time_feature = data_normalize(time_feature)


def generate_training_set(file):
    while 1:
        f = open(file)
        for line in f:
            time_id = int(line)
            # create Numpy arrays of input data and labels, from each time_id
            x = np.zeros((link_num, input_dim), dtype=np.float)
            y = np.zeros((link_num, 1), dtype=np.float)
            for i in range(link_num):
                y[i, :] = event_traffic[link_list[i]][time_id]

                cur_ind = 0
                x[i, cur_ind: cur_ind+coarse_feature_dim] = poi_type_feature_coarse[i, :]
                cur_ind += coarse_feature_dim
                x[i, cur_ind: cur_ind + fine_feature_dim] = poi_type_feature_fine[i, :]
                cur_ind += fine_feature_dim
                x[i, cur_ind: cur_ind + link_info_feature_dim] = link_info_feature[i, :]
                cur_ind += link_info_feature_dim
                x[i, cur_ind: cur_ind + time_feature_dim] = time_feature[time_id, :]
                cur_ind += time_feature_dim
            yield (x, y)
        f.close()


def generate_testing_set(file):
    while 1:
        f = open(file)
        for line in f:
            time_id = int(line)
            # create Numpy arrays of input data and labels, from each time_id
            x = np.zeros((link_num, input_dim), dtype=np.float)
            y = np.zeros((link_num, 1), dtype=np.float)
            for i in range(link_num):
                y[i, :] = event_traffic[link_list[i]][time_id]

                cur_ind = 0
                x[i, cur_ind: cur_ind+coarse_feature_dim] = poi_type_feature_coarse[i, :]
                cur_ind += coarse_feature_dim
                x[i, cur_ind: cur_ind + fine_feature_dim] = poi_type_feature_fine[i, :]
                cur_ind += fine_feature_dim
                x[i, cur_ind: cur_ind + link_info_feature_dim] = link_info_feature[i, :]
                cur_ind += link_info_feature_dim
                x[i, cur_ind: cur_ind + time_feature_dim] = time_feature[time_id, :]
                cur_ind += time_feature_dim
            yield (x, y)
        f.close()


model = Sequential()
model.add(Dense(1, input_dim=input_dim))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

model.fit_generator(generate_training_set(root_dir+"train_time_id.txt"), steps_per_epoch=train_time, epochs=10,
                    validation_data=generate_testing_set(root_dir+"test_time_id.txt"), validation_steps=test_time)
