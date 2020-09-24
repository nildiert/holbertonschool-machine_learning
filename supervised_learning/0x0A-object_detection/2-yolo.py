#!/usr/bin/env python3
"""
Yolo v3 algorithm to perform object detection
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    This class uses the Yolo v3 algorithm
    to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializer method
        """
        class_names = []
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _sigmoid(self, x):
        """
        Auxiliar sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process outputs
        """

        boxes = []

        for i, output in enumerate(outputs):

            grid_h, grid_w, num_boxes, _ = output.shape
            image_height, image_width = image_size

            anchors_size = self.anchors.reshape(1, 1,
                                                self.anchors.shape[0],
                                                num_boxes, 2)

            b_xy = self._sigmoid(output[:, :, :, :2])

            b_wh = np.exp(output[:, :, :, 2:4])
            b_wh *= anchors_size[:, :, i, :, :]

            c_x = np.tile(np.arange(
                0, grid_w), grid_w).reshape(grid_w, grid_w)
            c_y = np.tile(np.arange(
                0, grid_h), grid_w).reshape(grid_w, grid_h).T

            c_x = c_x.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)
            c_y = c_y.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)

            c_xy = np.concatenate((c_x, c_y), axis=3)

            b_xy += c_xy
            b_xy /= (grid_w, grid_h)
            b_wh /= (self.model.input.shape[1].value,
                     self.model.input.shape[2].value)
            b_xy -= (b_wh / 2)

            box = np.concatenate((b_xy, b_xy + b_wh), axis=-1)

            box[..., 0] *= image_width
            box[..., 1] *= image_height
            box[..., 2] *= image_width
            box[..., 3] *= image_height

            boxes.append(box)

        box_confidence = [self._sigmoid(out[..., 4:5]) for out in outputs]
        box_prob = [self._sigmoid(out[..., 5:]) for out in outputs]

        return (boxes, box_confidence, box_prob)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter Boxes
        """

        box_scores = [bc * bcp for bc, bcp in zip(box_confidences,
                                                  box_class_probs)]
        box_classes = []
        box_classes_scores = []
        for box_score in box_scores:

            box = np.argmax(box_score, axis=-1)
            box = box.flatten()
            box_classes.append(box)

            box_class_score = np.max(box_score, axis=-1)
            box_class_score = box_class_score.flatten()
            box_classes_scores.append(box_class_score)

        boxes = [box.reshape(-1, 4) for box in boxes]

        box_classes = np.concatenate(box_classes, axis=-1)
        box_classes_scores = np.concatenate(box_classes_scores, axis=-1)
        boxes = np.concatenate(boxes, axis=0)

        filt = np.where(box_classes_scores >= self.class_t)

        boxes = boxes[filt]
        box_classes = box_classes[filt]
        scores = box_classes_scores[filt]

        return (boxes, box_classes, scores)
