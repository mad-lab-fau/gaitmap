import numpy as np

def compare_label_lists_with_tolerance(ground_truth_labels, predicted_labels, tolerance=0):
    """Compare ground-truth label list with predicted label list with the given tolerance.

    This function will retrun a list of all true-positive, all false-positive and all false-negative labels.


    Parameters
    ----------
    ground_truth_labels : list of labels or 2D array
        Ground truth list of label pairs [start, end] which define one stride.

    predicted_labels : list of labels or 2D array
        Predicted list of label pairs [start, end] which define one stride.

    tolerance: int
        Deviation between compared values in samples, after values are still considered a match (e.g. [2,4] and [1,6]
        would still count as a match with given tolerance = 2.

    Returns :
    -------
    [true_positive_labels, false_positive_labels, false_negative_labels]
        retruns a list of all true-positive, all false-positive and all false-negative labels


    Examples
    --------
    >>> ground_truth_labels = [[10,20],[21,30],[31,40],[50,60]]
    >>> predicted_labels = [[10,21],[20,34],[31,40]]
    >>> tp_labels, fp_labels, fn_labels = compare_label_lists_with_tolerance(ground_truth_labels, predicted_labels, tolerance = 2)
    >>> tp_labels
    >>> ... array([[10, 21], [31, 40]])
    >>> fp_labels
    >>> ... array([20, 34])
    >>> false_negative_labels
    >>> ... array([[21, 30], [50, 60]])

    """
    if np.array(ground_truth_labels).size == 0:
        false_positive_labels = np.array(predicted_labels)
        if false_positive_labels.ndim == 1 and false_positive_labels.size > 0:
            false_positive_labels = false_positive_labels.reshape(1, 2)
        return np.empty(0), false_positive_labels, np.empty(0)

    if np.array(predicted_labels).size == 0:
        false_negative_labels = np.array(ground_truth_labels)
        if false_negative_labels.ndim == 1 and false_negative_labels.size > 0:
            false_negative_labels = false_negative_labels.reshape(1, 2)
        return np.empty(0), np.empty(0), false_negative_labels

    similar = np.zeros(np.shape(ground_truth_labels))
    for l in predicted_labels:
        similar = np.logical_or(similar, np.abs(ground_truth_labels - np.asarray(l)) <= tolerance)
    bool_similar = np.array([np.array_equal([True, True], t) for t in similar])

    idx_false = np.squeeze(np.argwhere(~bool_similar).tolist())
    if idx_false.size == 0:
        false_negative_labels = np.empty(0)
    else:
        false_negative_labels = np.take(ground_truth_labels, idx_false, axis=0)
        if false_negative_labels.ndim == 1:
            false_negative_labels = false_negative_labels.reshape(1, 2)

    # get true positive and false positive labels by comparing all ground truth labels with predicted label list
    similar = np.zeros(np.shape(predicted_labels))
    for l in ground_truth_labels:
        similar = np.logical_or(similar, np.abs(predicted_labels - np.asarray(l)) <= tolerance)
    bool_similar = np.array([np.array_equal([True, True], t) for t in similar])

    idx_true = np.squeeze(np.argwhere(bool_similar).tolist())
    if idx_true.size == 0:
        true_positive_labels = np.empty(0)
    else:
        true_positive_labels = np.take(predicted_labels, idx_true, axis=0)
        if true_positive_labels.ndim == 1:
            true_positive_labels = true_positive_labels.reshape(1, 2)

    idx_false = np.squeeze(np.argwhere(~bool_similar).tolist())
    if idx_false.size == 0:
        false_positive_labels = np.empty(0)
    else:
        false_positive_labels = np.take(predicted_labels, idx_false, axis=0)
        if false_positive_labels.ndim == 1:
            false_positive_labels = false_positive_labels.reshape(1, 2)

    return true_positive_labels, false_positive_labels, false_negative_labels