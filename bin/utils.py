from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np
import json

def cal_metric(y_true, y_pred, ignore_index=9):
    """Calculates the Matthews correlation coefficient for binary classification.

    Parameters:
        - y_true (array): True binary labels.
        - y_pred (array): Predicted binary labels.

    Returns:
        - float: Matthews
    """
    binary_classification = y_pred.shape[-1] == 2
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    label = y_true.reshape(-1)
    predict = y_pred.reshape(-1)

    mask = label != ignore_index
    label = label[mask]
    predict = predict[mask]
    if binary_classification:
        result = {
            'mcc_score': matthews_corrcoef(label, predict),
            'f1_score': f1_score(label, predict),
            'accuracy_score': accuracy_score(label, predict),
            'recall_score': recall_score(label, predict),
            'precision_score': precision_score(label, predict),
            # 'roc_auc_score': roc_auc_score(label, predict),
        }
    else:
        # fix sum up to 1.0 over classes
        # pred_prob = np.exp(pred_prob) / np.sum(np.exp(pred_prob), axis=1, keepdims=True)
        result = {
            'mcc_score': matthews_corrcoef(label, predict),
            'f1_score': f1_score(label, predict, average='macro'),
            'accuracy_score': accuracy_score(label, predict),
            'recall_score': recall_score(label, predict, average='macro'),
            'precision_score': precision_score(label, predict, average='macro'),
        }
    print(f"label: {label}, max label: {label.max()}")
    confu_matrix = confusion_matrix(label, predict)
    if confu_matrix.shape[-1] > 20:
        confu_matrix = confu_matrix[:20, :20]
        print("Confusion matrix is too large, only show the first 9x9 part")
    result["confu_matrix"] = str(confu_matrix)
    print(f"eval metric: \n{json.dumps(result)}")