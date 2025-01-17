# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 40000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA2.0'
CLASS_NUM = 18

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA2.0_1x_20210813'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta)) + 180, sin^2(theta) + cos^2(theta) = 1
[-90, 90]   sin in [-1, 1]   cos in [0, 1]

This is your evaluation result for task 1:

mAP: 0.4458437614121817
ap of each class: plane:0.7402263224804869, baseball-diamond:0.49505805445369216, bridge:0.37282365114297317, ground-track-field:0.5952399740955576, small-vehicle:0.34390443732599546, large-vehicle:0.3338629984397782, ship:0.43972789183769895, tennis-court:0.7591685915175856, basketball-court:0.5549993068684814, storage-tank:0.5118234164554336, soccer-ball-field:0.40638376772093465, roundabout:0.4928485005968652, harbor:0.3388100093701085, swimming-pool:0.529949566929558, helicopter:0.5133102432120433, container-crane:0.09530877285979326, airport:0.38610577443831995, helipad:0.11563642567396436
The submitted information is :

Description: RetinaNet_DOTA2.0_1x_20210813_52w_v1
Username: sjtu-deter
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""



