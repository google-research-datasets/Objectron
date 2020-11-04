"""Parser for Objectron tf.examples."""

import math
import numpy as np
import tensorflow as tf
import cv2

import objectron.schema.features as features

# Label names.
LABEL_INSTANCE = '2d_instance'
LABEL_INSTANCE_3D = '3d_instance'
VISIBILITY = 'visibility'

class ObjectronParser(object):
  """Parser using NumPy."""

  def __init__(self, height = 640, width = 480):
    self._in_height, self._in_width = height, width
    self._vis_thresh = 0.1
    
  def get_image_and_label(self, serialized):
    """Gets image and its label from a serialized tf.Example.

    Args:
      serialized: A string of serialized tf.Example.

    Returns:
      A tuple of image and its label.
    """
    example = tf.train.Example.FromString(serialized)
    return self.parse_example(example)

  def parse_example(self, example):
    """Parses image and label from a tf.Example proto.

    Args:
      example: A tf.Example proto.

    Returns:
      A tuple of image and its label.
    """
    fm = example.features.feature
    image = self.get_image(
        fm[features.FEATURE_NAMES['IMAGE_ENCODED']], shape=(self._in_width, self._in_height))
    image = image / 255.
    image = self._normalize_image(image)

    label = {}
    visibilities = fm[features.FEATURE_NAMES['VISIBILITY']].float_list.value
    visibilities = np.asarray(visibilities)
    label[VISIBILITY] = visibilities
    index = visibilities > self._vis_thresh

    if features.FEATURE_NAMES['POINT_2D'] in fm:
      points_2d = fm[features.FEATURE_NAMES['POINT_2D']].float_list.value
      points_2d = np.asarray(points_2d).reshape((-1, 9, 3))[..., :2]
      label[LABEL_INSTANCE] = points_2d[index]

    if features.FEATURE_NAMES['POINT_3D'] in fm:
      points_3d = fm[features.FEATURE_NAMES['POINT_3D']].float_list.value
      points_3d = np.asarray(points_3d).reshape((-1, 9, 3))
      label[LABEL_INSTANCE_3D] = points_3d[index]

    return image, label

  def parse_camera(self, example):
    """Parses camera from a tensorflow example."""
    fm = example.features.feature
    if features.FEATURE_NAMES['PROJECTION_MATRIX'] in fm:
      proj = fm[features.FEATURE_NAMES['PROJECTION_MATRIX']].float_list.value
      proj = np.asarray(proj).reshape((4, 4))
    else:
      proj = None

    if features.FEATURE_NAMES['VIEW_MATRIX'] in fm:
      view = fm[features.FEATURE_NAMES['VIEW_MATRIX']].float_list.value
      view = np.asarray(view).reshape((4, 4))
    else:
      view = None
    return proj, view

  def parse_plane(self, example):
    """Parses plane from a tensorflow example."""
    fm = example.features.feature
    if features.FEATURE_NAMES['PLANE_CENTER'] in fm and features.FEATURE_NAMES['PLANE_NORMAL'] in fm:
      center = fm[features.FEATURE_NAMES['PLANE_CENTER']].float_list.value
      center = np.asarray(center)
      normal = fm[features.FEATURE_NAMES['PLANE_NORMAL']].float_list.value
      normal = np.asarray(normal)
      return center, normal
    else:
      return None

  def get_image(self, feature, shape=None):
    image = cv2.imdecode(
        np.asarray(bytearray(feature.bytes_list.value[0]), dtype=np.uint8),
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(image.shape) > 2 and image.shape[2] > 1:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if shape is not None:
      image = cv2.resize(image, shape)
    return image

  def _normalize_image(self, image):
    """Normalizes pixels of an image from [0, 1] to [-1, 1]."""
    return image * 2. - 1.

