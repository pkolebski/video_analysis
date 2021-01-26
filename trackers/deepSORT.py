from typing import List

import numpy as np
import cv2
from utils.download import download_url
from trackers.tracker import BaseTracker
from detectors.detector import Detection
import torch
from trackers.deep_sort import nn_matching
from trackers.deep_sort.detectiondeepsort import DetectionDeepSORT
from trackers.deep_sort.tracker import Tracker
from trackers.deep_sort import generate_detections as gdet
import torchvision

PATH_MODEL = "pretrained_models"

from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.stats import multivariate_normal


def get_gaussian_mask():
    # 128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.22, 0.22])
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask


class DeepSORT(BaseTracker):
    def __init__(self,
                 model_url="https://github.com/abhyantrika/nanonets_object_tracking/blob/master/ckpts/model640.pt?raw=true",
                 model_filename = PATH_MODEL+"/model640.pt"):
        # super(DeepSORT, self).__init__()
        self.active_tracked = dict()
        self.inactive_tracked = dict()
        max_cosine_distance = 0.3
        nn_budget = None
        x ="model640.pt"
        #if model_filename.split(".")[-1] == "pb":
        #    self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        download_url(model_url, model_filename)
        self.encoder = torch.load(model_filename)
        self.encoder = self.encoder.cuda()
        self.encoder = self.encoder.eval()
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=30)
        self.history = {}
        self.transforms = torchvision.transforms.Compose([ \
            torchvision.transforms.ToPILImage(), \
            torchvision.transforms.Resize((128, 128)), \
            torchvision.transforms.ToTensor()])
        self.gaussian_mask = get_gaussian_mask().cuda()

    def pre_process(self, frame, detections):

        transforms = torchvision.transforms.Compose([ \
            torchvision.transforms.ToPILImage(), \
            torchvision.transforms.Resize((128, 128)), \
            torchvision.transforms.ToTensor()])

        crops = []
        for d in detections:

            for i in range(len(d)):
                if d[i] < 0:
                    d[i] = 0

            img_h, img_w, img_ch = frame.shape

            xmin, ymin, w, h = d

            if xmin > img_w:
                xmin = img_w

            if ymin > img_h:
                ymin = img_h

            xmax = xmin + w
            ymax = ymin + h

            ymin = abs(int(ymin))
            ymax = abs(int(ymax))
            xmin = abs(int(xmin))
            xmax = abs(int(xmax))

            try:
                crop = frame[ymin:ymax, xmin:xmax, :]
                crop = transforms(crop)
                crops.append(crop)
            except:
                continue

        crops = torch.stack(crops)

        return crops

    def extract_features_only(self, frame, coords):

        for i in range(len(coords)):
            if coords[i] < 0:
                coords[i] = 0

        img_h, img_w, img_ch = frame.shape

        xmin, ymin, w, h = coords

        if xmin > img_w:
            xmin = img_w

        if ymin > img_h:
            ymin = img_h

        xmax = xmin + w
        ymax = ymin + h

        ymin = abs(int(ymin))
        ymax = abs(int(ymax))
        xmin = abs(int(xmin))
        xmax = abs(int(xmax))

        crop = frame[ymin:ymax, xmin:xmax, :]
        # crop = crop.astype(np.uint8)

        # print(crop.shape,[xmin,ymin,xmax,ymax],frame.shape)

        crop = self.transforms(crop)
        crop = crop.cuda()

        gaussian_mask = self.gaussian_mask

        input_ = crop * gaussian_mask
        input_ = torch.unsqueeze(input_, 0)

        features = self.encoder.forward_once(input_)
        features = features.detach().cpu().numpy()

        corrected_crop = [xmin, ymin, xmax, ymax]

        return features, corrected_crop

    def match_bbs(self, bbs: List[Detection], frame) -> None:
        for k, v in self.active_tracked.items():
            v[1] = False

        boxes = np.array([(detection.position[0], detection.position[1], detection.position[2] - detection.position[0],
                           detection.position[3] - detection.position[1]) for detection in bbs])

        ##(x, y, width, height)
        names = np.array([detection.obj_type for detection in bbs])
        scores = np.array([detection.probability for detection in bbs])
        processed_crops = self.pre_process(frame, boxes).cuda()
        processed_crops = self.gaussian_mask * processed_crops
        # features = np.array(self.encoder(frame, boxes))
        features = self.encoder.forward_once(processed_crops)
        features = features.detach().cpu().numpy()
        if len(features.shape) == 1:
            features = np.expand_dims(features, 0)

        detections = [DetectionDeepSORT(bbox, score, class_name, feature)
                      for bbox, score, class_name, feature in
                      zip(boxes, scores, names, features)]

        self.tracker.predict()
        self.tracker.update(detections)

        tracked_bboxes = []
        tracked_ids = []
        for track in self.tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            ##(min x, min y, max x, max y)
            tracking_id = track.track_id
            tracked_ids.append(tracking_id)
            class_name = track.get_class()
            # detection = Detection(position=bbox, obj_type=class_name, probability=0)
            if tracking_id in self.history.keys():
                self.history[tracking_id].append(Detection(position=bbox, obj_type=class_name, probability=0))
            else:
                self.history[tracking_id] = [Detection(position=bbox, obj_type=class_name, probability=0)]
            self.active_tracked[tracking_id] = [0, True, self.history[tracking_id]]

        # for tracks_id in self.active_tracked.keys():
        #    if tracks_id not in tracked_ids:
        #        self.active_tracked[tracks_id][2] = False
        self.inactive_tracked = {k: v for k, v in self.active_tracked.items() if not v[1]}
        self.active_tracked = {k: v for k, v in self.active_tracked.items() if v[1]}
