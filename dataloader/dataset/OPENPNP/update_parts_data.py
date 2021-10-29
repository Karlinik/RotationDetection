import os
import sys
import json
import numpy as np

MICRO_CRYSTAL = "Micro Crystal"
INFINEON = "Infineon"
NORDIC_SEMICONDUCTOR = "Nordic Semiconductor"
U_BLOX = "u-blox"
STMICROELECTRONICS = "STMicroelectronics"

# data_dir = 'C:\\Users\\Nikola\\Documents\\Skola\\diplomka\\r3det\\R3Det_Tensorflow\\data\\io\\OPENPNP\\dataset\\val\\labelJson'
data_dir = 'C:\\Users\\Nikola\\.openpnp2\\parts_dataset'
jsons = [i for i in os.listdir(data_dir) if 'json' in i]

parts = {}

for idx, data in enumerate(jsons):
    # load the data
    json_file = open(os.path.join(data_dir, data), 'r')
    json_data = json.load(json_file)
    json_file.close()

    # alter the data 
    name = json_data['part']['name']

    if name == MICRO_CRYSTAL:
        json_data['part']['size'] = [3.2, 1.5]
    
    if name == INFINEON:
        json_data['part']['size'] = [0.7, 1.1]

    # write the data
    json_file = open(os.path.join(data_dir, data), 'w')
    json.dump(json_data, json_file)
    json_file.close()

print(parts)

# {'Micro Crystal': [3.2, 3.2], 'Infineon': [1.1, 1.1], 'Nordic Semiconductor': [5, 5], 'u-blox': [4.5, 4.5], 'STMicroelectronics': [2, 2]}
# {'Micro Crystal': [3.2, 1.5], 'Infineon': [0.7, 1.1], 'Nordic Semiconductor': [5, 5], 'u-blox': [4.5, 4.5], 'STMicroelectronics': [2, 2]}