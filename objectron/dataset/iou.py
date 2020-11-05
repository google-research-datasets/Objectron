"""The Intersection Over Union (IoU) for 3D oriented bounding boxes."""

import numpy as np
import scipy.spatial as sp

import objectron.dataset.box as Box

_PLANE_THICKNESS_EPSILON = 0.000001
_POINT_IN_FRONT_OF_PLANE = 1
_POINT_ON_PLANE = 0
_POINT_BEHIND_PLANE = -1


class IoU(object):
  """General Intersection Over Union cost for Oriented 3D bounding boxes."""

  def __init__(self, box1, box2):
    self._box1 = box1
    self._box2 = box2
    self._intersection_points = []

  def iou(self):
    """Computes the exact IoU using Sutherland-Hodgman algorithm."""
    self._intersection_points = []
    self._compute_intersection_points(self._box1, self._box2)
    self._compute_intersection_points(self._box2, self._box1)
    if self._intersection_points:
      intersection_volume = sp.ConvexHull(self._intersection_points).volume
      box1_volume = self._box1.volume
      box2_volume = self._box2.volume
      union_volume = box1_volume + box2_volume - intersection_volume
      return intersection_volume / union_volume
    else:
      return 0.

  def iou_sampling(self, num_samples=10000):
    """Computes intersection over union by sampling points.

    Generate n samples inside each box and check if those samples are inside
    the other box. Each box has a different volume, therefore the number o
    samples in box1 is estimating a different volume than box2. To address
    this issue, we normalize the iou estimation based on the ratio of the
    volume of the two boxes.

    Args:
      num_samples: Number of generated samples in each box

    Returns:
      IoU Estimate (float)
    """
    p1 = [self._box1.sample() for _ in range(num_samples)]
    p2 = [self._box2.sample() for _ in range(num_samples)]
    box1_volume = self._box1.volume
    box2_volume = self._box2.volume
    box1_intersection_estimate = 0
    box2_intersection_estimate = 0
    for point in p1:
      if self._box2.inside(point):
        box1_intersection_estimate += 1
    for point in p2:
      if self._box1.inside(point):
        box2_intersection_estimate += 1
    # We are counting the volume of intersection twice.
    intersection_volume_estimate = (
        box1_volume * box1_intersection_estimate +
        box2_volume * box2_intersection_estimate) / 2.0
    union_volume_estimate = (box1_volume * num_samples + box2_volume *
                             num_samples) - intersection_volume_estimate
    iou_estimate = intersection_volume_estimate / union_volume_estimate
    return iou_estimate

  def _compute_intersection_points(self, box_src, box_template):
    """Computes the intersection of two boxes."""
    # Transform the source box to be axis-aligned
    inv_transform = np.linalg.inv(box_src.transformation)
    box_src_axis_aligned = box_src.apply_transformation(inv_transform)
    template_in_src_coord = box_template.apply_transformation(inv_transform)
    for face in range(len(Box.FACES)):
      indices = Box.FACES[face, :]
      poly = [template_in_src_coord.vertices[indices[i], :] for i in range(4)]
      clip = self.intersect_box_poly(box_src_axis_aligned, poly)
      for point in clip:
        # Transform the intersection point back to the world coordinate
        point_w = np.matmul(box_src.rotation, point) + box_src.translation
        self._intersection_points.append(point_w)

    for point_id in range(Box.NUM_KEYPOINTS):
      v = template_in_src_coord.vertices[point_id, :]
      if box_src_axis_aligned.inside(v):
        point_w = np.matmul(box_src.rotation, v) + box_src.translation
        self._intersection_points.append(point_w)

  def intersect_box_poly(self, box, poly):
    """Clips the polygon against the faces of the axis-aligned box."""
    for axis in range(3):
      poly = self._clip_poly(poly, box.vertices[1, :], 1.0, axis)
      poly = self._clip_poly(poly, box.vertices[8, :], -1.0, axis)
    return poly

  def _clip_poly(self, poly, plane, normal, axis):
    """Clips the polygon with the plane using the Sutherland-Hodgman algorithm.

    See en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm for the overview of
    the Sutherland-Hodgman algorithm. Here we adopted a robust implementation
    from "Real-Time Collision Detection", by Christer Ericson, page 370.

    Args:
      poly: List of 3D vertices defining the polygon.
      plane: The 3D vertices of the (2D) axis-aligned plane.
      normal: normal
      axis: A tuple defining a 2D axis.

    Returns:
      List of 3D vertices of the clipped polygon.
    """
    # The vertices of the clipped polygon are stored in the result list.
    result = []
    if len(poly) <= 1:
      return result

    # polygon is fully located on clipping plane
    poly_in_plane = True

    # Test all the edges in the polygon against the clipping plane.
    for i, current_poly_point in enumerate(poly):
      prev_poly_point = poly[(i + len(poly) - 1) % len(poly)]
      d1 = self._classify_point_to_plane(prev_poly_point, plane, normal, axis)
      d2 = self._classify_point_to_plane(current_poly_point, plane, normal,
                                         axis)
      if d2 == _POINT_BEHIND_PLANE:
        poly_in_plane = False
        if d1 == _POINT_IN_FRONT_OF_PLANE:
          intersection = self._intersect(plane, prev_poly_point,
                                         current_poly_point, axis)
          result.append(intersection)
        elif d1 == _POINT_ON_PLANE:
          if not result or (not np.array_equal(result[-1], prev_poly_point)):
            result.append(prev_poly_point)
      elif d2 == _POINT_IN_FRONT_OF_PLANE:
        poly_in_plane = False
        if d1 == _POINT_BEHIND_PLANE:
          intersection = self._intersect(plane, prev_poly_point,
                                         current_poly_point, axis)
          result.append(intersection)
        elif d1 == _POINT_ON_PLANE:
          if not result or (not np.array_equal(result[-1], prev_poly_point)):
            result.append(prev_poly_point)

        result.append(current_poly_point)
      else:
        if d1 != _POINT_ON_PLANE:
          result.append(current_poly_point)

    if poly_in_plane:
      return poly
    else:
      return result

  def _intersect(self, plane, prev_point, current_point, axis):
    """Computes the intersection of a line with an axis-aligned plane.

    Args:
      plane: Formulated as two 3D points on the plane.
      prev_point: The point on the edge of the line.
      current_point: The other end of the line.
      axis: A tuple defining a 2D axis.

    Returns:
      A 3D point intersection of the poly edge with the plane.
    """
    alpha = (current_point[axis] - plane[axis]) / (
        current_point[axis] - prev_point[axis])
    # Compute the intersecting points using linear interpolation (lerp)
    intersection_point = alpha * prev_point + (1.0 - alpha) * current_point
    return intersection_point

  def _inside(self, plane, point, axis):
    """Check whether a given point is on a 2D plane."""
    # Cross products to determine the side of the plane the point lie.
    x, y = axis
    u = plane[0] - point
    v = plane[1] - point

    a = u[x] * v[y]
    b = u[y] * v[x]
    return a >= b

  def _classify_point_to_plane(self, point, plane, normal, axis):
    """Classify position of a point w.r.t the given plane.

    See Real-Time Collision Detection, by Christer Ericson, page 364.

    Args:
      point: 3x1 vector indicating the point
      plane: 3x1 vector indicating a point on the plane
      normal: scalar (+1, or -1) indicating the normal to the vector
      axis: scalar (0, 1, or 2) indicating the xyz axis

    Returns:
      Side: which side of the plane the point is located.
    """
    signed_distance = normal * (point[axis] - plane[axis])
    if signed_distance > _PLANE_THICKNESS_EPSILON:
      return _POINT_IN_FRONT_OF_PLANE
    elif signed_distance < -_PLANE_THICKNESS_EPSILON:
      return _POINT_BEHIND_PLANE
    else:
      return _POINT_ON_PLANE

  @property
  def intersection_points(self):
    return self._intersection_points