import os
from xml.dom.minidom import Document
import numpy as np
import copy
import cv2
import sys
import json
sys.path.insert(0, 'C:\\Users\\Nikola\\Documents\\Skola\\diplomka\\r3det\\RotationDetection')
# sys.path.append('../../..')

# from libs.box_utils.coordinate_convert import backward_convert, forward_convert
from libs.utils.coordinate_convert import backward_convert, forward_convert


def save_to_xml(save_path, im_height, im_width, objects_axis, label_name):
    im_depth = 0
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('OPENPNP')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_name = doc.createTextNode('000024.jpg')
    filename.appendChild(filename_name)
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('The OPENPNP custom Dataset'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('XML'))
    source.appendChild(annotation_s)

    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)

    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('xxxxxxxx'))
    source.appendChild(flickrid)

    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('xxxxxxxx'))
    owner.appendChild(flickrid_o)

    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('karlinik'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(im_width)))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(im_height)))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(im_depth)))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(object_num):
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(label_name[int(objects_axis[i][-1])]))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        x0 = doc.createElement('x0')
        x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
        bndbox.appendChild(x0)
        y0 = doc.createElement('y0')
        y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
        bndbox.appendChild(y0)

        x1 = doc.createElement('x1')
        x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
        bndbox.appendChild(x1)
        y1 = doc.createElement('y1')
        y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
        bndbox.appendChild(y1)

        x2 = doc.createElement('x2')
        x2.appendChild(doc.createTextNode(str((objects_axis[i][4]))))
        bndbox.appendChild(x2)
        y2 = doc.createElement('y2')
        y2.appendChild(doc.createTextNode(str((objects_axis[i][5]))))
        bndbox.appendChild(y2)

        x3 = doc.createElement('x3')
        x3.appendChild(doc.createTextNode(str((objects_axis[i][6]))))
        bndbox.appendChild(x3)
        y3 = doc.createElement('y3')
        y3.appendChild(doc.createTextNode(str((objects_axis[i][7]))))
        bndbox.appendChild(y3)

    f = open(save_path, 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()

class_list = ['part', 'nozzle']

def format_label(json_file, x_c, y_c):
    format_data = []
    input = json.load(json_file)

    # x_c = x_c - 45
    # y_c = y_c - 45

    # center of the box
    units_x, units_y = input['unitsPerPixel']

    x_dev = input['XDeviation']//units_x
    y_dev = input['YDeviation']//units_y

    format_data.extend([x_c + x_dev, y_c - y_dev])

    # part width and height - TODO: find units
    part_w = input['part']['size'][0]/units_x
    part_h = input['part']['size'][1]/units_y
    format_data.extend([part_w, part_h])

    # Theta
    rotation = input['RDeviation']
    if rotation > 0:
        format_data.append((-1)*rotation)
    else:
        format_data.append((-90)-rotation)

    # class
    format_data.append(class_list.index('part'))

    return np.array([format_data])

def draw_box(image, box, box5):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    box = np.int0(box[:8].reshape([4,2]))
    print(box)

    cv2.drawContours(image,[box],0,(0,0,255),6)
    height = image.shape[0]
    width = image.shape[1]
    image = cv2.circle(image, (width//2, (height-45)//2), radius=0, color=(150, 150, 0), thickness=20)
    image = cv2.circle(image, (int(box5[0][0]), int(box5[0][1])), radius=0, color=(0, 0, 250), thickness=20)
    image = cv2.line(image, (0, 0), (width//2, 0), (0, 255, 0), thickness=20)
    image = cv2.line(image, (0, 0), (0, (height-45)//2), (0, 255, 0), thickness=20)
    image = cv2.line(image, (0, height), (0, height-45), (0, 255, 0), thickness=20)
  
    cv2.imshow("output", image) 
    cv2.waitKey(0)

def clip_image(file_idx, image, img_box, stride_w, stride_h):
    min_pixel = 2
    print(file_idx)
    # To draw a rectangle, we need 4 corners of the rectangle -> OpenCV provides function cv2.boxPoints(). 
    # This takes as input the Box2D structure and returns the 4 corner points.
    # Box2D structure contains: (center(x, y), (width, height), angle of rotation)

    box_all = forward_convert(img_box)
    print(box_all[np.logical_or(img_box[:, 2] <= min_pixel, img_box[:, 3] <= min_pixel), :])
    box_all = box_all[np.logical_and(img_box[:, 2] > min_pixel, img_box[:, 3] > min_pixel), :]

    # draw_box(image, box_all[0], img_box)

    if box_all.shape[0] > 0:
        shape = image.shape
        for start_h in range(0, shape[0], stride_h):
            for start_w in range(0, shape[1], stride_w):
                boxes = copy.deepcopy(box_all)
                box = np.zeros_like(box_all)
                top_left_row = 0
                top_left_col = 0
                bottom_right_row = shape[0]
                bottom_right_col = shape[1]

                subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]

                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col
                box[:, 4] = boxes[:, 4] - top_left_col
                box[:, 6] = boxes[:, 6] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                box[:, 5] = boxes[:, 5] - top_left_row
                box[:, 7] = boxes[:, 7] - top_left_row
                box[:, 8] = boxes[:, 8]
                center_y = 0.25 * (box[:, 1] + box[:, 3] + box[:, 5] + box[:, 7])
                center_x = 0.25 * (box[:, 0] + box[:, 2] + box[:, 4] + box[:, 6])

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)
                if len(idx) > 0 and (subImage.shape[0] > 5 and subImage.shape[1] > 5):
                    if not os.path.exists(os.path.join(save_dir, 'images')):
                        print("path doesn't exist, creating dirs: ", os.path.join(save_dir, 'images'))
                        os.makedirs(os.path.join(save_dir, 'images'))
                    img = os.path.join(save_dir, 'images',
                                       "%s_%04d_%04d.png" % (file_idx, top_left_row, top_left_col))
                    cv2.imwrite(img, subImage)

                    if not os.path.exists(os.path.join(save_dir, 'labeltxt')):
                        os.mkdir(os.path.join(save_dir, 'labeltxt'))
                    xml = os.path.join(save_dir, 'labeltxt',
                                       "%s_%04d_%04d.xml" % (file_idx, top_left_row, top_left_col))
                    save_to_xml(xml, subImage.shape[0], subImage.shape[1], box[idx, :], class_list)

print('class_list', len(class_list))
raw_data = '.\\data\\dataset\\OPENPNP\\val_test'
# raw_data = '/data/dataset/OPENPNP/val_test/'
raw_images_dir = os.path.join(raw_data, 'images')
raw_label_dir = os.path.join(raw_data, 'labelJson')

# save_dir = '\\data\\dataset\\OPENPNP\\trainval\\'
# save_dir = '/data/dataset/OPENPNP/trainval/'
save_dir = 'C:\\Users\\Nikola\\Documents\\Skola\\diplomka\\r3det\\RotationDetection\\data\\dataset\\OPENPNP\\trainval_test\\'

images = [i for i in os.listdir(raw_images_dir) if 'jpeg' in i]
jsons = [i for i in os.listdir(raw_label_dir) if 'json' in i]

print('find image', len(images))
print('find label', len(jsons))

## find out how much can I crop the image
stride_h, stride_w = 450, 450

for idx, img in enumerate(images):
    # print(idx, 'read image', img)
    img_data = cv2.imread(os.path.join(raw_images_dir, img))
    height, width, l = img_data.shape

    json_data = open(os.path.join(raw_label_dir, img.replace('jpeg', 'json')), 'r')
    box = format_label(json_data, width//2, height//2)

    if box.shape[0] > 0:
        clip_image(img.strip('.jpeg'), img_data, box, stride_w, stride_h)
