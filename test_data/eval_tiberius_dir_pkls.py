import glob
import json
import os
import pickle
import numpy as np
import tqdm
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, recall_score, precision_score, confusion_matrix


def tiberius_reduce_labels(y_batch: np.ndarray, output_size=7):
    """
    Set output size for the model with tiberius labels, only support y_batch dim are 7 and 15
    :param output_size:
    :return:
    """
    # reformat labels so that they fit the output size
    if y_batch.shape[-1] == 7:
        if output_size == 5:
            # reduce intron labels
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = y_batch[..., 0]
            y_new[..., 1] = np.sum(y_batch[..., 1:4], axis=-1)
            y_new[..., 2:] = y_batch[..., 4:]
        elif output_size == 3:
            # reduce intron and exon labels
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = y_batch[..., 0]
            y_new[..., 1] = np.sum(y_batch[..., 1:4], axis=-1)
            y_new[..., 2] = np.sum(y_batch[..., 4:], axis=-1)
        elif output_size == 15:
            # reduce intron and exon labels
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., :y_batch.shape[-1]] = y_batch
        else:
            y_new = y_batch.astype(np.float32)
        y_batch = y_new
    elif y_batch.shape[-1] == 15:
        if output_size == 3:
            # reduce intron and exon labels
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = y_batch[..., 0]
            y_new[..., 1] = np.sum(y_batch[..., 1:4], axis=-1)
            y_new[..., 2] = np.sum(y_batch[..., 4:], axis=-1)
        elif output_size == 5:
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = y_batch[..., 0]
            y_new[..., 1] = np.sum(y_batch[..., 1:4], axis=-1)
            y_new[..., 2] = np.sum(y_batch[..., [4, 7, 10, 12]], axis=-1)
            y_new[..., 3] = np.sum(y_batch[..., [5, 8, 13]], axis=-1)
            y_new[..., 4] = np.sum(y_batch[..., [6, 9, 11, 14]], axis=-1)
        elif output_size == 7:
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., :4] = y_batch[..., :4]
            y_new[..., 4] = np.sum(y_batch[..., [4, 7, 10, 12]], axis=-1)
            y_new[..., 5] = np.sum(y_batch[..., [5, 8, 13]], axis=-1)
            y_new[..., 6] = np.sum(y_batch[..., [6, 9, 11, 14]], axis=-1)
        elif output_size == 15:
            y_new = y_batch.astype(np.float32)
        elif output_size == 2:
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = np.sum(y_batch[..., :4], axis=-1)
            y_new[..., 1] = np.sum(y_batch[..., 4:], axis=-1)
        elif output_size == 4:
            y_new = np.zeros(list(y_batch.shape[:-1]) + [output_size], np.float32)
            y_new[..., 0] = np.sum(y_batch[..., :4], axis=-1)
            y_new[..., 1] = np.sum(y_batch[..., [4, 7, 10, 12]], axis=-1)
            y_new[..., 2] = np.sum(y_batch[..., [5, 8, 13]], axis=-1)
            y_new[..., 3] = np.sum(y_batch[..., [6, 9, 11, 14]], axis=-1)
        else:
            raise ValueError(f"output size {output_size} not supported, only support 7, 15 classes")
        y_batch = y_new
    return y_batch


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
    print("confusion matrix shape: \n", confu_matrix.shape)
    if confu_matrix.shape[-1] > 20:
        confu_matrix = confu_matrix[:20, :20]
        print("Confusion matrix is too large, only show the first 9x9 part")
    result["confu_matrix"] = str(confu_matrix)
    print(f"eval metric: \n{json.dumps(result)}")


def eval_dir_pkls(pkl_dir, num_classes=15):
    pkl_files = glob.glob(os.path.join(pkl_dir, '*.pkl'))

    y_predicts = []
    y_trues = []
    for pkl_file in tqdm.tqdm(pkl_files):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            y_true = data['y_true']
            y_true_onehot = np.eye(num_classes)[y_true]
            y_predict = data['y_pred']
            y_predict_onehot = np.eye(num_classes)[y_predict]
            y_trues.append(y_true_onehot)
            y_predicts.append(y_predict_onehot)

    y_trues = np.concatenate(y_trues, axis=0)
    y_trues = tiberius_reduce_labels(y_trues, 7)
    y_predicts = np.concatenate(y_predicts, axis=0)
    y_predicts = tiberius_reduce_labels(y_predicts, 7)
    cal_metric(y_trues, y_predicts)


def parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', type=str, default='.')
    parser.add_argument('--num_classes', type=int, default=15)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    eval_dir_pkls(pkl_dir=args.pkl_dir, num_classes=args.num_classes)
