import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    TP = sum([1 if prediction[i] == True and ground_truth[i] == True else 0 for i in range(len(prediction))])
    TN = sum([1 if prediction[i] == False and ground_truth[i] == False else 0 for i in range(len(prediction))])
    FP = sum([1 if prediction[i] == True and ground_truth[i] == False else 0 for i in range(len(prediction))])
    FN = sum([1 if prediction[i] == False and ground_truth[i] == True else 0 for i in range(len(prediction))])
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    recall = TP / (TP + FN)
    
    precision = TP / (TP + FP)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    return sum(np.equal(prediction, ground_truth)) / len(ground_truth)
