#!/usr/bin/env python3
"""
Yolo v3 algorithm to perform object detection
"""

import tensorflow.keras as K
import numpy as np
import glob
import cv2
import os


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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Non max suppression
        """
        n_boxes, n_classes, n_scores = [], [], []

        for classes in set(box_classes):
            index = np.where(box_classes == classes)
            boxes = filtered_boxes[index]
            class_val = box_classes[index]
            scores = box_scores[index]

            x1, x2 = boxes[:, 0], boxes[:, 2]
            y1, y2 = boxes[:, 1], boxes[:, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idx = scores.argsort()[::-1]
            keep = []

            while idx.size > 0:
                ii = idx[0]
                jj = idx[1:]
                keep.append(ii)

                xx1 = np.maximum(x1[ii], x1[jj])
                yy1 = np.maximum(y1[ii], y1[jj])
                xx2 = np.minimum(x2[ii], x2[jj])
                yy2 = np.minimum(y2[ii], y2[jj])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)

                box_area = (w * h)
                overlap = box_area / (area[ii] + area[jj] - box_area)
                idxs = np.where(overlap <= self.nms_t)[0]
                idx = idx[idxs + 1]

            keep = np.array(keep)

            n_boxes.append(boxes[keep])
            n_classes.append(class_val[keep])
            n_scores.append(scores[keep])

        predict_boxes = np.concatenate(n_boxes)
        predict_classes = np.concatenate(n_classes)
        predict_scores = np.concatenate(n_scores)

        return (predict_boxes, predict_classes, predict_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Load images
        """
        images = []
        paths_images = glob.glob(folder_path + '/*')

        for path in paths_images:
            img = cv2.imread(path)
            images.append(img)
        return (images, paths_images)

    def preprocess_images(self, images):
        """
        Preprocess images
        """

        dimensions = []
        resize_images = []
        input_h = self.model.input.shape[2].value
        input_w = self.model.input.shape[1].value
        for image in images:
            dimensions.append(image.shape[:2])

        dimensions = np.stack(dimensions, axis=0)

        newSize = (input_w, input_h)

        for image in images:
            resize_img = cv2.resize(image, newSize,
                                    interpolation=cv2.INTER_CUBIC)
            resize_img = resize_img / 255
            resize_images.append(resize_img)

        resize_images = np.stack(resize_images, axis=0)

        return (resize_images, dimensions)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Show predicted boxes
        """
        rounded_scores = np.around(box_scores, decimals=2)

        for i, box in enumerate(boxes):
            x1, y2, x2, y1 = box

            start = (int(x1), int(y1))
            end = (int(x2), int(y2))
            blue = (255, 0, 0)

            cv2.rectangle(image, start, end, blue, 2)

            score = str(rounded_scores[i])
            class_name = self.class_names[box_classes[i]]
            classname_score = "{} {}".format(class_name, score)
            start_text = (int(x1), int(y2)-5)
            font_name = cv2.FONT_HERSHEY_SIMPLEX
            red = (0, 0, 255)
            text_thickness = 1
            line_type = cv2.LINE_AA
            font_scale = 0.5

            cv2.putText(image,
                        classname_score,
                        start_text,
                        font_name,
                        font_scale,
                        red,
                        text_thickness,
                        line_type)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists("./detections"):
                os.makedirs("./detections")
            cv2.imwrite("./detections/{}".format(file_name), image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Predict function
        """
        pred = []
        images, image_paths = self.load_images(folder_path)
        process_imgs, image_shapes = self.preprocess_images(images)
        outputs = self.model.predict(process_imgs)

        for i, img in enumerate(images):
            outs = [outputs[0][i, :, :, :, :],
                    outputs[1][i, :, :, :, :],
                    outputs[2][i, :, :, :, :]]
            img_original_dimensions = np.array([img.shape[0],
                                                img.shape[1]])
            bxs, box_conf, box_cl_probs = self.process_outputs(
                outs,
                img_original_dimensions)
            boxes, box_classes, box_scores = self.filter_boxes(
                bxs,
                box_conf,
                box_cl_probs)
            boxes, box_classes, box_scores = self.non_max_suppression(
                boxes,
                box_classes,
                box_scores)
            image_name = image_paths[i].split('/')[-1]
            self.show_boxes(img,
                            boxes,
                            box_classes,
                            box_scores,
                            image_name)
            pred.append((boxes, box_classes, box_scores))

        return (pred, image_paths)
