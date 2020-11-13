"""Features in Objectron's tf.Example."""

import frozendict
import tensorflow as tf


# Single view feature names.
FEATURE_NAMES = frozendict.frozendict({
    # An encoded image in an example.
    'IMAGE_ENCODED': 'image/encoded',
    # The path and the filename for the image in the local filesystem
    'IMAGE_FILENAME': 'image/filename',
    # Image format.
    'IMAGE_FORMAT': 'image/format',
    # Frame number in the sequence.
    'IMAGE_ID': 'image/id',
    'IMAGE_WIDTH': 'image/width',
    'IMAGE_HEIGHT': 'image/height',
    # The segmentation mask (encoded image).
    'IMAGE_ALPHA': 'image/alpha',
    # The color space of the image (e.g. RGB, BGR, RGBA, Grayscale, etc.)
    'COLOR_SPACE': 'image/colorspace',
    # The number of channels in the image
    'NUM_CHANNELS': 'image/channels',
    # The microsecond timestamp of the video frame in the video stream.
    'TIMESTAMP_MCSEC': 'image/timestamp',
    # A list of float numbers of all 2D points in an example.
    'POINT_2D': 'point_2d',
    # Similar with above but for 3D points.
    'POINT_3D': 'point_3d',
    # A list of point numbers for each instance. The bounding box
    # has 9 points, Skeleton may have a varied number of points.
    'POINT_NUM': 'point_num',
    # Number of object instances in this frame
    'INSTANCE_NUM': 'instance_num',
    # A row major 4x4 projection matrix
    'PROJECTION_MATRIX': 'camera/projection',
    # Similar with above, but for view matrices.
    'VIEW_MATRIX': 'camera/view',
    # A row major 4x4 transformation matrix describing the camera pose w.r.t.
    # the world origin. The world origin is where the AR session has started.
    'EXTRINSIC_MATRIX': 'camera/extrinsics',
    # A row-major 3x3 intrinsic matrix describing the focal length and the
    # principal point of the camera.
    'INTRINSIC_MATRIX': 'camera/intrinsics',
    # A string indicating the orientation of the camera (portrait, landscape).
    'ORIENTATION': 'camera/orientation',
    # A list of object names in this frame.
    'OBJECT_NAME': 'object/name',
    # A list of object translations.
    'OBJECT_TRANSLATION': 'object/translation',
    # A list of object orientation.
    'OBJECT_ORIENTATION': 'object/orientation',
    # A list of object scales.
    'OBJECT_SCALE': 'object/scale',
    # A list of annotation visibilities
    'VISIBILITY': 'object/visibility',
    # The center point for the ground plane objects are sitting on.
    'PLANE_CENTER': 'plane/center',
    # The normal vector for the ground plane objects are sitting on.
    'PLANE_NORMAL': 'plane/normal',
})

FEATURE_MAP = {
    FEATURE_NAMES['IMAGE_ENCODED']:
        tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
    FEATURE_NAMES['IMAGE_FORMAT']:
        tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value='png'),
    FEATURE_NAMES['IMAGE_FILENAME']:
        tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
    FEATURE_NAMES['IMAGE_WIDTH']:
        tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0),
    FEATURE_NAMES['IMAGE_HEIGHT']:
        tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0),
    FEATURE_NAMES['IMAGE_ALPHA']:
        tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
    FEATURE_NAMES['COLOR_SPACE']:
        tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value='RGB'),
    FEATURE_NAMES['NUM_CHANNELS']:
        tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=3),
    FEATURE_NAMES['TIMESTAMP_MCSEC']:
        tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
    FEATURE_NAMES['IMAGE_ID']:
        tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=-1),
    FEATURE_NAMES['POINT_2D']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['POINT_3D']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['POINT_NUM']:
        tf.io.VarLenFeature(dtype=tf.int64),
    FEATURE_NAMES['INSTANCE_NUM']:
        tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0),
    FEATURE_NAMES['PROJECTION_MATRIX']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['VIEW_MATRIX']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['INTRINSIC_MATRIX']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['EXTRINSIC_MATRIX']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['ORIENTATION']:
        tf.io.VarLenFeature(dtype=tf.string),
    FEATURE_NAMES['OBJECT_NAME']:
        tf.io.VarLenFeature(dtype=tf.string),
    FEATURE_NAMES['OBJECT_TRANSLATION']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['OBJECT_ORIENTATION']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['OBJECT_SCALE']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['VISIBILITY']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['PLANE_CENTER']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['PLANE_NORMAL']:
        tf.io.VarLenFeature(dtype=tf.float32),
}

# This is pretty much identical to the FEATURE_MAP for single frame examples, 
# the only difference is instead of FixedLenFeature, each feature has the type FixedLenSequenceFeature.
# VarLenFeature remains the same.
SEQUENCE_FEATURE_MAP = {
    FEATURE_NAMES['IMAGE_ENCODED']:
        tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string),
    FEATURE_NAMES['IMAGE_FORMAT']:
        tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string),
    FEATURE_NAMES['IMAGE_FILENAME']:
        tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string),
    FEATURE_NAMES['IMAGE_WIDTH']:
        tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64),
    FEATURE_NAMES['IMAGE_HEIGHT']:
        tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64),
    FEATURE_NAMES['COLOR_SPACE']:
        tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string),
    FEATURE_NAMES['NUM_CHANNELS']:
        tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64),
    FEATURE_NAMES['TIMESTAMP_MCSEC']:
        tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
    FEATURE_NAMES['IMAGE_ID']:
        tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64),
    FEATURE_NAMES['POINT_2D']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['POINT_3D']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['POINT_NUM']:
        tf.io.VarLenFeature(dtype=tf.int64),
    FEATURE_NAMES['INSTANCE_NUM']:
        tf.io.FixedLenSequenceFeature(shape=[1], dtype=tf.int64),
    FEATURE_NAMES['PROJECTION_MATRIX']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['VIEW_MATRIX']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['INTRINSIC_MATRIX']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['EXTRINSIC_MATRIX']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['ORIENTATION']:
        tf.io.VarLenFeature(dtype=tf.string),
    FEATURE_NAMES['OBJECT_NAME']:
        tf.io.VarLenFeature(dtype=tf.string),
    FEATURE_NAMES['OBJECT_TRANSLATION']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['OBJECT_ORIENTATION']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['OBJECT_SCALE']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['VISIBILITY']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['PLANE_CENTER']:
        tf.io.VarLenFeature(dtype=tf.float32),
    FEATURE_NAMES['PLANE_NORMAL']:
        tf.io.VarLenFeature(dtype=tf.float32),
}

SEQUENCE_CONTEXT_MAP = {
    'sequence_id':
        tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
    'count':
        tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1),
}