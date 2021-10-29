import os
from xml.dom.minidom import Document
import numpy as np
import copy
import cv2
import sys
import json
sys.path.insert(0, 'C:\\Users\\Nikola\\Documents\\Skola\\diplomka\\r3det\\R3Det_Tensorflow')

from libs.box_utils.coordinate_convert import backward_convert, forward_convert

class_list = ['part', 'nozzle']
images_coco = []
annotations = []
annotation_id = 0

def format_label(input, x_c, y_c):
    min_pixel = 5
    format_data = []

    # x_c = x_c - 45
    # y_c = y_c - 45

    units_x, units_y = input['unitsPerPixel']

    x_dev = input['XDeviation']//units_x
    y_dev = input['YDeviation']//units_y

    format_data.extend([x_c + x_dev, y_c - y_dev])

    part_w = input['part']['size'][0]/units_x
    part_h = input['part']['size'][1]/units_y
    # format_data.extend([part_w, part_h])
    # format_data.extend([part_h, part_w])

    rotation = input['RDeviation']
    if rotation > 0:
        format_data.extend([part_w, part_h])
        format_data.append((-1)*rotation)
    else:
        format_data.extend([part_h, part_w])
        format_data.append((-90)-rotation)

    format_data.append(class_list.index('part'))

    box = np.array([format_data])
    box_all = forward_convert(box)
    # print(box_all[np.logical_or(box[:, 2] <= min_pixel, box[:, 3] <= min_pixel), :])
    box_all = box_all[np.logical_and(box[:, 2] > min_pixel, box[:, 3] > min_pixel), :]

    return box[0], box_all[0]

def draw_box(image, box_all, box):
    print(box_all)
    print(box)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    box_all = np.int0(box_all[:8].reshape([4,2]))
    print(box_all)

    cv2.drawContours(image,[box_all],0,(0,0,255),6)
    height = image.shape[0]
    width = image.shape[1]
    image = cv2.circle(image, (width//2, (height-45)//2), radius=0, color=(150, 150, 0), thickness=20)
    image = cv2.circle(image, (int(box[0]), int(box[1])), radius=0, color=(0, 0, 250), thickness=20)
    image = cv2.line(image, (0, 0), (width//2, 0), (0, 255, 0), thickness=20)
    image = cv2.line(image, (0, 0), (0, (height-45)//2), (0, 255, 0), thickness=20)
    image = cv2.line(image, (0, height), (0, height-45), (0, 255, 0), thickness=20)
  
    cv2.imshow("output", image) 
    cv2.waitKey(0)

def coco_convert(img_name, input, box, box_all):
    global annotation_id
    image = {}
    annotation = {}
    box_all = [round(x) for x in box_all]

    units_x, units_y = input['unitsPerPixel']
    part_w = input['part']['size'][0]/units_x
    part_h = input['part']['size'][1]/units_y
    img_id = img_name.strip('TestImage').strip('.jpeg')

    image['file_name'] = img_name
    image['height'] = round(part_h)
    image['width'] = round(part_w)
    image['id'] = img_id

    annotation['segmentation'] = [[]]
    annotation['area'] = int(part_h)*int(part_w)
    annotation['iscrowd'] = 0
    annotation['image_id'] = img_id
    annotation['bbox'] = [box_all[0], box_all[1], box_all[2], box_all[3]]
    annotation['category_id'] = box[-1]
    annotation['id'] = annotation_id

    annotations.append(annotation)
    images_coco.append(image)
    annotation_id = annotation_id + 1


print('class_list', len(class_list))
raw_data = 'C:\\Users\\Nikola\\Documents\\Skola\\diplomka\\r3det\\R3Det_Tensorflow\\data\\io\\OPENPNP\\dataset\\val_tmp'
raw_images_dir = os.path.join(raw_data, 'images')
raw_label_dir = os.path.join(raw_data, 'labelJson')

save_dir = 'C:\\Users\\Nikola\\Documents\\Skola\\diplomka\\r3det\\R3Det_Tensorflow\\data\\io\\OPENPNP\\dataset\\coco_format\\'

images = [i for i in os.listdir(raw_images_dir) if 'jpeg' in i]
jsons = [i for i in os.listdir(raw_label_dir) if 'json' in i]

print('find image', len(images))
print('find label', len(jsons))

# TODO: crop?

for idx, img in enumerate(images):
    # print(img)
    img_data = cv2.imread(os.path.join(raw_images_dir, img))
    height, width, l = img_data.shape

    json_data = open(os.path.join(raw_label_dir, img.replace('jpeg', 'json')), 'r')
    input = json.load(json_data)
    box, box_all = format_label(input, width//2, height//2)
    # print(box)
    # print(box_all)

    if box.shape[0] > 0:
        coco_convert(img, input, box, box_all)
        # draw_box(img_data, box_all, box)

categories = [{'id': class_list.index(x), 'name': x} for x in class_list]

# print(images_coco)
# print(annotations)
# print(categories)

coco_final = {}
coco_final['images'] = images_coco
coco_final['annotations'] = annotations

coco_final['categories'] = categories

with open("data\\io\\OPENPNP\\dataset\\coco.json", "w") as outfile:
    json.dump(coco_final, outfile)