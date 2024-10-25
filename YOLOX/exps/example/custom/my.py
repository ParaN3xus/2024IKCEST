#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/COCO"
        
        #self.train_ann = "instances_slim.json"
        #self.val_ann = "instances_slim.json"
        
        self.train_ann = "instances_train2017_3_classes.json"
        self.val_ann = "instances_val2017_3_classes.json"

        self.num_classes = 3

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
        self.max_epoch = 300
