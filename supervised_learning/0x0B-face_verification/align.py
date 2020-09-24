#!/usr/bin/env python3
"""
FaceAlign Class
"""

import dlib
import numpy as np
import cv2


class FaceAlign:
    """
    FaceAlign class
    """

    def __init__(self, shape_predictor_path):
        """ Constructor method of the class
        Arg:
            shape_predictor_path: is the path to the dlib shape predictor model
        Sets the public instance attributes:
            detector - contains dlib‘s default face detector
            shape_predictor - contains the dlib.shape_predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """ Detects a face in an image:
        Arg:
            image: is a numpy.ndarray of rank 3 containing an image
                   from which to detect a face
        Returns:
            dlib.rectangle containing the boundary box for the face in the
            image, or None on failure
            - If multiple faces are detected, return the dlib.rectangle
              with the largest area
            - If no faces are detected, return a dlib.rectangle that is the
            same as the image
        """
        try:
            # Find faces in the image(s)
            faces = self.detector(image, 1)
            max = 0
            # Not face detected
            if len(faces) == 0:
                # dlib.rectangle(left, top, right, bottom)
                # rectangle is a square with the size of the image
                rectangle = dlib.rectangle(0,
                                           0,
                                           image.shape[1],
                                           image.shape[0])
            # one or more faces founded
            if len(faces) >= 1:
                for i, face in enumerate(faces):
                    # Find the largest rectangle
                    if face.area() > max:
                        max, rectangle = face.area(), face

            return rectangle
        except RuntimeError:
            return None

    def find_landmarks(self, image, detection):
        """ Finds facial landmarks
        Args:
            image: is a numpy.ndarray of an image from which to find facial
            landmarks
            detection: is a dlib.rectangle containing the boundary box of
            the face in the image
        Returns:
            Returns: a numpy.ndarray of shape (p, 2)containing the landmark
            points, or None on failure
            - p is the number of landmark points
            2 is the x and y coordinates of the point
        """
        # Find landmark points in image
        landmark = self.shape_predictor(image, detection)

        # If landmark is founded, create an array with the coordinates
        if landmark:
            coordinates = np.zeros((68, 2), dtype="int")
            for i in range(68):
                x, y = landmark.part(i).x, landmark.part(i).y
                coordinates[i] = [x, y]
        # If not found landmarks return None
        else:
            coordinates = None

        return coordinates

    def align(self, image, landmark_indices, anchor_points, size=96):
        """ Aligns an image for face verification
        Args:
            image:  is a numpy.ndarray of rank 3 containing the image to
            be aligned
            landmark_indices: is a numpy.ndarray of shape (3,) containing
                the indices of the three landmark points that should be
                used for the affine transformation
            anchor_points:  is a numpy.ndarray of shape (3, 2) containing
                the destination points for the affine transformation, scaled
                to the range [0, 1]
            size: is the desired size of the aligned image
        Returns:
            a numpy.ndarray of shape (size, size, 3) containing the aligned
                image, or None if no face
            is detected
        """
        # Detect face in image and find landmarks
        box = self.detect(image)
        landmarks = self.find_landmarks(image, box)

        # Select three points in the landmarks(Eyes and nose)
        points_in_image = landmarks[landmark_indices]
        points_in_image = points_in_image.astype('float32')
        # Generate the normalized output size
        output_size = anchor_points * size

        # Calculates the 2 \times 3 matrix of an affine transform
        # cv2.getAffineTransform(src, dst)
        # src – Coordinates of triangle vertices in the source image.
        # dst – Coordinates of the corresponding triangle vertices in the
        # destination image.
        affine_transf = cv2.getAffineTransform(points_in_image, output_size)

        # Transforms the source image using the specified matrix
        # cv2.warpAffine(src, M, dsize[, dst[, flags[]]])
        # src – input image
        # dst – output image that has the size dsize and the same type as src
        # M – 2\times 3 transformation matrix
        transformed_image = cv2.warpAffine(image, affine_transf, (size, size))

        return transformed_image
