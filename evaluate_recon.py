from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def evaluate_metrics(real_labels, pred_labels):
    f1 = f1_score(real_labels, pred_labels)
    precision = precision_score(real_labels, pred_labels)
    recall = recall_score(real_labels, pred_labels)
    accuracy = accuracy_score(real_labels, pred_labels)
    print(f"F1 Score: {round(f1, 4)}, Precision: {round(precision, 4)}, Recall: {round(recall, 4)}, Accuracy: {round(accuracy, 4)}")
    return f1, precision, recall, accuracy

def confusion_matrix_metrics(real_labels, pred_labels):
    tn, fp, fn, tp = confusion_matrix(y_true = real_labels, y_pred = pred_labels).ravel().tolist()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    return tn, fp, fn, tp