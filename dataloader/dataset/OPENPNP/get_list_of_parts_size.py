import os
import sys
import json
import numpy as np

data_dir = 'C:\\Users\\Nikola\\.openpnp2\\parts_dataset'
jsons = [i for i in os.listdir(data_dir) if 'json' in i]

parts = {}

for idx, data in enumerate(jsons):
    json_file = open(os.path.join(data_dir, data), 'r')
    json_data = json.load(json_file)

    name = json_data['part']['name']
    if name not in parts:
        parts[name] = json_data['part']['size']
    elif parts[name] != json_data['part']['size']:
        parts[name].append(json_data['part']['size'])

print(parts)

# {'Micro Crystal': [3.2, 3.2], 'Infineon': [1.1, 1.1], 'Nordic Semiconductor': [5, 5], 'u-blox': [4.5, 4.5], 'STMicroelectronics': [2, 2]}
# {'Micro Crystal': [3.2, 1.5], 'Infineon': [0.7, 1.1], 'Nordic Semiconductor': [5, 5], 'u-blox': [4.5, 4.5], 'STMicroelectronics': [2, 2]}