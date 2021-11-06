import os
from xml.dom.minidom import Document
import numpy as np
import copy
import cv2
import sys
import json
sys.path.insert(0, 'C:\\Users\\Nikola\\Documents\\Skola\\diplomka\\r3det\\RotationDetection')

def json_bb_convert(img_name, input, x_c, y_c):
    units_x, units_y = input['unitsPerPixel']

    x_dev = input['XDeviation']/units_x
    y_dev = input['YDeviation']/units_y

    bounding_box = {}
    bounding_box['center'] = [x_c + x_dev, y_c - y_dev]
    bounding_box['angle'] = input['RDeviation']

    annotation = {}
    annotation['file_name'] = img_name
    annotation['bounding_box'] = bounding_box
    annotation['category_id'] = 0

    save_dir = "data\\dataset\\OPENPNP\\sanity\\"

    if not os.path.exists(save_dir):
        print("path doesn't exist, creating dirs: ", save_dir)
        os.makedirs(save_dir)

    json_name = img_name.strip('.jpeg') + '.json'
    print(json_name)
    with open(save_dir + json_name, "w") as outfile:
        json.dump(annotation, outfile)


raw_data = 'C:\\Users\\Nikola\\Documents\\Skola\\diplomka\\r3det\\RotationDetection\\data\\dataset\\OPENPNP\\val_test'
raw_images_dir = os.path.join(raw_data, 'images')
raw_label_dir = os.path.join(raw_data, 'labelJson')

images = [i for i in os.listdir(raw_images_dir) if 'jpeg' in i]
jsons = [i for i in os.listdir(raw_label_dir) if 'json' in i]

print('find image', len(images))
print('find label', len(jsons))

for idx, img in enumerate(images):
    img_data = cv2.imread(os.path.join(raw_images_dir, img))
    height, width, l = img_data.shape

    json_data = open(os.path.join(raw_label_dir, img.replace('jpeg', 'json')), 'r')
    input = json.load(json_data)
    json_bb_convert(img, input, width/2, height/2)
        