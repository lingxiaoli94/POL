import torch
import numpy as np
import math
from .universal_solver_base import UniversalSolverBase
import pol.problems.objdetect
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def convert_frcnn_box(boxes, width, height):
    result = []
    for box in boxes:
        x0, y0, x1, y1 = box
        x0 = x0 / width
        y0 = y0 / height
        x1 = x1 / width
        y1 = y1 / height

        x = (x0 + x1) / 2
        y = (y0 + y1) / 2
        w = x1 - x0
        h = y1 - y0
        result.append([x, y, w, h])
    return result

def filter_by_score(boxes, scores, threshold):
    result = []
    for i in range(len(boxes)):
        if scores[i] >= threshold:
            result.append((boxes[i], scores[i]))
    result = sorted(result, key=lambda pr: pr[1], reverse=True)
    result = [pr[0] for pr in result]

    return result

'''
Special solver for objdetect that uses Faster-RCNN.
'''

class FRCNN(UniversalSolverBase, torch.nn.Module):
    def __init__(self, *, device, threshold, max_num_box):
        super().__init__()
        self.device = device
        self.threshold = threshold
        self.max_num_box = max_num_box

        self.model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
        self.model.to(self.device)
        self.model.eval()

    def set_problem(self, problem):
        super().set_problem(problem)
        assert(isinstance(problem, pol.problems.objdetect.ObjectDetection))

    def predict_boxes(self, data_batch):
        batch_img = data_batch['img_ori'] / 255.0
        pred = self.model(batch_img)

        result_boxes = []
        for i, img in enumerate(batch_img):
            boxes = pred[i]['boxes'].detach()
            if len(boxes) == 0:
                boxes = [[0, 0, 1, 1]]
                scores = [0.0]
            else:
                boxes = convert_frcnn_box(boxes, width=img.shape[1], height=img.shape[2])
                scores = pred[i]['scores']
            filtered_boxes = filter_by_score(boxes, scores, threshold=self.threshold)
            if len(filtered_boxes) > self.max_num_box:
                filtered_boxes = filtered_boxes[:self.max_num_box]
            if len(filtered_boxes) == 0:
                filtered_boxes = [boxes[0]]
            first_box = filtered_boxes[0]
            if len(filtered_boxes) < self.max_num_box:
                filtered_boxes.extend(
                    [first_box] * (self.max_num_box - len(filtered_boxes)))
            result_boxes.append(torch.as_tensor(filtered_boxes))
        result_boxes = torch.stack(result_boxes, 0).to(self.device)
        assert(result_boxes.shape[1] == self.max_num_box and
               result_boxes.shape[2] == 4)

        return result_boxes

    def compute_losses(self, data_batch, prior_samples):
        # Should never train this.
        assert(False)
