import numpy as np

'''
Sample-based metrics for comparing full system behaviors 
'''

'''
finds all true negatives, true positives, false negatives, false positives

true positive - both voiced & equal pitch
true negative - both unvoiced
false positive - both voiced + wrong pitch or unvoiced annotation vs voiced estimation
false negative - voiced annotation vs non-voiced estimation
'''


def find_cases(annotations, estimations):
    tn = np.isnan(annotations) & np.isnan(estimations)
    tp = np.equal(annotations, estimations)
    fn = (~np.isnan(annotations) & np.isnan(estimations))
    fp_1 = np.isnan(annotations) & ~np.isnan(estimations)
    fp_2 = ~np.isnan(annotations) & ~np.isnan(estimations) & np.not_equal(annotations, estimations)
    fp = fp_1 | fp_2
    return tn, tp, fn, fp


'''
accuracy, precision, recall , f-score
'''


def calculate_metrics(annotations, estimations):
    tn, tp, fn, fp = find_cases(annotations, estimations)
    tp_n = np.count_nonzero(tp)
    tn_n = np.count_nonzero(tn)
    fp_n = np.count_nonzero(fp)
    fn_n = np.count_nonzero(fp_n)
    precision = 0
    recall = 0
    acc = 0
    f_score = 0
    if tp_n + fp_n != 0:
        precision = tp_n / (tp_n + fp_n)
    if tp_n + fn_n != 0:
        recall = tp_n / (tp_n + fn_n)
    acc = (tp_n + tn_n) / (tp_n + tn_n + fp_n + fn_n)
    if precision + recall != 0:
        f_score = 2 * precision * recall / (precision + recall)
    return acc, precision, recall, f_score
