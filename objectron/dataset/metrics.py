"""Util classes for computing evaluation metrics."""

import numpy as np


class HitMiss(object):
  """Class for recording hits and misses of detection results."""

  def __init__(self, thresholds):
    self.thresholds = thresholds
    self.size = thresholds.shape[0]
    self.hit = np.zeros(self.size, dtype=np.float)
    self.miss = np.zeros(self.size, dtype=np.float)

  def reset(self):
    self.hit = np.zeros(self.size, dtype=np.float)
    self.miss = np.zeros(self.size, dtype=np.float)

  def record_hit_miss(self, metric, greater=True):
    """Records the hit or miss for the object based on the metric threshold."""
    for i in range(self.size):
      threshold = self.thresholds[i]
      hit = (greater and metric >= threshold) or (
          (not greater) and metric <= threshold)
      if hit:
        self.hit[i] += 1
      else:
        self.miss[i] += 1


class AveragePrecision(object):
  """Class for computing average precision."""

  def __init__(self, size):
    self.size = size
    self.aps = np.zeros(size)
    self.true_positive = []
    self.false_positive = []
    for _ in range(size):
      self.true_positive.append([])
      self.false_positive.append([])
    self._total_instances = 0.

  def append(self, hit_miss, num_instances):
    for i in range(self.size):
      self.true_positive[i].append(hit_miss.hit[i])
      self.false_positive[i].append(hit_miss.miss[i])
    self._total_instances += num_instances

  def compute_ap(self, recall, precision):
    """Calculates the AP given the recall and precision array.

    The reference implementation is from Pascal VOC 2012 eval script. First we
    filter the precision recall rate so precision would be monotonically
    decreasing. Next, we compute the average precision by numerically
    integrating the precision-recall curve.

    Args:
      recall: Recall list
      precision: Precision list

    Returns:
      Average precision.
    """
    recall = np.insert(recall, 0, [0.])
    recall = np.append(recall, [1.])
    precision = np.insert(precision, 0, [0.])
    precision = np.append(precision, [0.])
    monotonic_precision = precision.copy()
    # Make the precision monotonically decreasing.
    for i in range(len(monotonic_precision) - 2, -1, -1):
      monotonic_precision[i] = max(monotonic_precision[i],
                                   monotonic_precision[i + 1])

    recall_changes = []
    for i in range(1, len(recall)):
      if recall[i] != recall[i - 1]:
        recall_changes.append(i)
    # Compute the average precision by integrating the recall curve.
    ap = 0.0
    for step in recall_changes:
      delta_recall = recall[step] - recall[step - 1]
      ap += delta_recall * monotonic_precision[step]
    return ap

  def compute_ap_curve(self):
    """Computes the precision/recall curve."""
    if self._total_instances == 0:
      raise ValueError('No instances in the computation.')

    for i in range(self.size):
      tp, fp = self.true_positive[i], self.false_positive[i]
      tp = np.cumsum(tp)
      fp = np.cumsum(fp)
      tp_fp = tp + fp
      recall = tp / self._total_instances
      precision = np.divide(tp, tp_fp, out=np.zeros_like(tp), where=tp_fp != 0)
      self.aps[i] = self.compute_ap(recall, precision)


class Accuracy(object):
  """Class for accuracy metric."""

  def __init__(self):
    self._errors = []
    self.acc = []

  def add_error(self, error):
    """Adds an error."""
    self._errors.append(error)

  def compute_accuracy(self, thresh=0.1):
    """Computes accuracy for a given threshold."""
    if not self._errors:
      return 0
    return len(np.where(np.array(self._errors) <= thresh)[0]) * 100. / (
        len(self._errors))