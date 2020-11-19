import torch

def prediction(scores, k):
    idxs = torch.topk(scores, k).indices
    prediction = torch.zeros_like(scores)
    for idx in list(enumerate(idxs)):
        prediction[idx[0], idx[1]] = 1
    return prediction

def add_count(count, output, labels):

    tp = (labels * output).sum()
    tn = ((1 - labels) * (1 - output)).sum()
    fp = ((1 - labels) * output).sum()
    fn = (labels * (1 - output)).sum()

    count[0] += tp.item()
    count[1] += tn.item()
    count[2] += fp.item()
    count[3] += fn.item()

    return count


def f1_score(count):

    epsilon = 1e-7
    tp, tn, fp, fn = count
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2*(precision*recall) / (precision + recall + epsilon)

    return f1