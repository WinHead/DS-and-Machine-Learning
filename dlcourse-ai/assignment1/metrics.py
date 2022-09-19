def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
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
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    true_answers = sum([1 if prediction[i] == ground_truth[i] else 0 for i in range(len(prediction))])
    false_answers = len(prediction) - true_answers
    
    accuracy = true_answers / (true_answers + false_answers)
    
    return accuracy
