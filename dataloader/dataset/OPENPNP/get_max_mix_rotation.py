import os
import sys
import json
import numpy as np

data_dir = 'C:\\Users\\Nikola\\.openpnp2\\parts_dataset'
jsons = [i for i in os.listdir(data_dir) if 'json' in i]

max_rotation = 0
min_rotation = 0

for idx, data in enumerate(jsons):
    json_file = open(os.path.join(data_dir, data), 'r')
    json_data = json.load(json_file)

    rotation = json_data['RDeviation']
    if rotation > max_rotation:
        max_rotation = rotation
    
    if rotation < min_rotation:
        min_rotation = rotation

    if rotation > 0 and rotation < 0.5:
        print("close to zero rotation file: ", data, " -> ", rotation)

print("MAX: ", max_rotation)
print("MIN: ", min_rotation)