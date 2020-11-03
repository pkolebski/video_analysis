import os
import cv2
import pandas as pd
from tqdm import tqdm
from utils.download import download_url

DATASET_PATH = 'data/MIO-TCD-Localization/'

objects = {'articulated_truck':0,
           'bicycle':1,
           'bus':2,
           'car':3,
           'motorcycle':4,
           'motorized_vehicle':5,
           'non-motorized_vehicle':6,
           'pedestrian':7,
           'pickup_truck':8,
           'single_unit_truck':9,
           'work_van':10}

if not os.path.isdir(DATASET_PATH):
    raise Exception('Dataset not found. Download dataset from http://podoce.dinf.usherbrooke.ca/static/dataset/MIO-TCD-Localization.tar \n'
                    'and unpack in /data')

header_list = ['image_id', "object", "x1", "y1", "x2", "y2"]
data = pd.read_csv(DATASET_PATH+"gt_train.csv", names=header_list)


def get_w_h(filename):
    image= cv2.imread(filename)
    shape = image.shape[:2]
    return shape[1], shape[0]

#00091639

def gen_txts():
    for file in tqdm(os.listdir(DATASET_PATH+"train")):
        if file.split('.')[-1] == 'jpg':
            file_id = int(file.split('.')[0])
            objects_img = data[data['image_id'] == file_id]
            w, h = get_w_h(DATASET_PATH+"train/"+file)
            f = open(DATASET_PATH+'train/'+str(file_id).zfill(8)+".txt", "w")
            tmp_str = ""
            for index, row in objects_img.iterrows():
                center_x = (row['x1'] + row['x2']) / 2
                center_y = (row['y1'] + row['y2']) / 2
                width = abs(row['x2'] - row['x1']) / w
                height = abs(row['y2'] - row['y1']) / h
                tmp_str=tmp_str+"{} {} {} {} {}\n".format(objects[row['object']], center_x/w, center_y/h, width, height)
            tmp_str = tmp_str[:-1]
            f.write(tmp_str)
            f.close()


def gen_test_train_txts():
    f = open(DATASET_PATH+"train.txt", 'w')
    iter = 0
    i = 0
    while iter < 100000:
        path = DATASET_PATH+"train/"+str(i).zfill(8)+".jpg"
        if os.path.isfile(path):
            f.write(path+"\n")
            iter = iter+1
        i = i + 1
    f.close()
    iter = 0
    f = open(DATASET_PATH+"test.txt", 'w')
    while iter < 10000:
        path = DATASET_PATH + "train/" + str(i).zfill(8) + ".jpg"
        if os.path.isfile(path):
            f.write(path + "\n")
            iter = iter + 1
        i = i + 1
    f.close()

download_url("https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29", "yolo_v4_train_params/yolov4-tiny.conv.29")
#gen_txts()
gen_test_train_txts()
