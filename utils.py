import csv
import torch

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    targets = targets.type(torch.LongTensor).cuda()
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size, pred
