import glob
import logging
import pickle
import random
import sys, os, json, csv, argparse

import pandas as pd

sys.path.append(".")
sys.path.append("./bin")
sys.path.append("/home/gabriell/gene_pred_deepl/bin")
sys.path.append("/home/gabriell/programs/learnMSA")
sys.path.append("/home/jovyan/brain//programs/learnMSA")

import tqdm
import numpy as np
from data_generator import DataGenerator
# from gene_pred_hmm import enePredHMMLayer
from t2t_pkl_dataset import T2TTiberiusDataset, T2TTiberiusTfrecordDataset, pytorch_to_tensorflow_dataset
from transformers import AutoTokenizer, TFAutoModelForMaskedLM, TFEsmForMaskedLM
from models import custom_cce_f1_loss
from utils import cal_metric, tiberius_reduce_labels
import tensorflow as tf
import tensorflow.keras as keras
from learnMSA.msa_hmm.Viterbi import viterbi


def read_species(file_name):
    """Reads a list of species from a given file, filtering out empty lines and comments.

    Parameters:
        - file_name (str): The path to the file containing species names.

        Returns:
        - list of str: A list of species names extracted from the file.
    """
    species = []
    with open(file_name, 'r') as f_h:
        species = f_h.read().strip().split('\n')
    return [s for s in species if s and s[0] != '#']


def decode_one_hot(encoded_seq):
    # Define the mapping from index to nucleotide
    index_to_nucleotide = np.array(['A', 'C', 'G', 'T', 'A'])
    # Use np.argmax to find the index of the maximum value in each row
    nucleotide_indices = np.argmax(encoded_seq, axis=-1)
    # Map indices to nucleotides
    decoded_seq = index_to_nucleotide[nucleotide_indices]
    # Convert from array of characters to string for each sequence
    decoded_seq_str = [''.join(seq) for seq in decoded_seq]
    return decoded_seq_str


def load_t2t_data_pkl(dest_path=None, batch_size=4, max_length=9999, dataset_name="gene_finding", split="train"):
    # load bend data
    t2t_dataset = T2TTiberiusTfrecordDataset(
        dest_path=dest_path,
        split=split,
        dataset_name=dataset_name,
        max_length=max_length,
        read_strand=False,
    )
    print("t2t_dataset", len(t2t_dataset))
    bend_dataset = pytorch_to_tensorflow_dataset(t2t_dataset)
    bend_dataset = bend_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return bend_dataset


def load_t2t_data_tfrecord(dest_path, batch_size=4, output_size=7):
    val_species = "val_species_filtered.txt"
    val_species = read_species(f'{dest_path}/{val_species}')
    test_file_path = []
    for specie in val_species:
        test_file_path.extend(glob.glob(f'{dest_path}/{specie}_*.tfrecords'))
    logging.info(f'Found {len(test_file_path)} val files.\n{test_file_path}')
    # max_test_file_num = 2
    # if len(test_file_path) > max_test_file_num:
    #     test_file_path = random.sample(test_file_path, max_test_file_num)
    #     logging.info(f'Using only 3 val files by random selected: {test_file_path}')

    val_data = DataGenerator(
        file_path=test_file_path,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        # max_nums=1000,
        output_size=output_size,
        hmm_factor=0,
    )
    return val_data


def eval_model(model, val_data, save_path=None, output_size: int = 7, save_prefix="val"):
    logging.info("Evaluating model on validation data")
    labels = []
    features = []
    y_predicts = []
    for i, val_i_data in tqdm.tqdm(enumerate(val_data), desc="evaluating"):
        feature, label = val_i_data
        y_predict = model.predict(feature)
        # get class of predict
        y_predict_label = np.argmax(y_predict, axis=-1)
        y_predict_onehot = np.eye(y_predict.shape[-1])[y_predict_label]
        y_predicts.append(y_predict_onehot)

        labels.append(label)
        features.append(feature)
        onehot_label = np.argmax(label, axis=-1)
        print(f"i:{i}; predict shape: {y_predict.shape}, feature shape: {feature.shape}, label shape: {label.shape}  "
              f"onehot_label: {onehot_label.flatten()[:10]}")
        # if i > 3:
        #     break
    y_predicts = np.concatenate(y_predicts, axis=0)
    labels = np.concatenate(labels, axis=0)
    features = np.concatenate(features, axis=0)
    y_catagories = np.argmax(labels, axis=-1)
    y_df = pd.DataFrame(y_catagories.flatten())
    #
    # save numpy result
    if save_path is not None:
        output_predict_label = tiberius_reduce_labels(y_predicts, output_size=3)
        save_name = f"{save_prefix}_tiberius_predict_pkl_result"
        np.savez(os.path.join(save_path, save_name),
                 labels=labels, predicts=y_predicts, features=features)
        result = {
            "labels": labels,
            "predicts": y_predicts,
            "features": features,
            "logits": output_predict_label
        }
        with open(os.path.join(save_path, f"{save_name}.pkl"), "wb") as f:
            pickle.dump(result, f)
    # data = np.load(os.path.join(save_path, "tiberius_predict_pkl_result.npz"))
    # labels = data["labels"]
    # y_predicts = data["predicts"]
    if y_predicts.shape[-1] > output_size:
        logging.info(f"reduce predicts from {y_predicts.shape[-1]} to {output_size}")
        y_predicts = tiberius_reduce_labels(y_predicts, output_size)
    if labels.shape[-1] > output_size:
        logging.info(f"reduce labels from {labels.shape[-1]} to {output_size}")
        labels = tiberius_reduce_labels(labels, output_size)

    print(f"label distribution each class: {y_df.value_counts()}")
    print(f"predicts shape: {y_predicts.shape}; label shape: {labels.shape}; catagories: {np.unique(y_catagories)}")
    print(f"predicts shape: {y_predicts.shape}; label shape: {labels.shape}")

    cal_metric(labels, y_predicts)


def load_val_data(file, hmm_factor=False, reduce_output=True, trans=True):
    data = np.load(file)
    x_val = data["array1"]
    y_val = data["array2"]
    print(f"x_val shape: {x_val.shape}; y_val shape: {y_val.shape}")

    if reduce_output:
        # reduce y_label size to 5
        y_new = np.zeros((y_val.shape[0], y_val.shape[1], 5), np.float32)
        y_new[:, :, 0] = y_val[:, :, 0]
        y_new[:, :, 1] = np.sum(y_val[:, :, 1:4], axis=-1)
        y_new[:, :, 2:] = y_val[:, :, 4:]
        y_val = y_new
    data.close()

    if hmm_factor:
        step_width = y_val.shape[1] // hmm_factor
        start = y_val[:, ::step_width, :]  # shape (batch_size, hmm_factor, 5)
        end = y_val[:, step_width - 1::step_width, :]  # shape (batch_size, hmm_factor, 5)
        hints = np.concatenate([start[:, :, tf.newaxis, :], end[:, :, tf.newaxis, :]], -2)
        return ([np.array(x_val), hints], [y_val, y_val])

    if trans:
        x_val = x_val[:, :99036]
        y_val = y_val[:, :99036]
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
        max_token_len = 5502
        x_token = np.reshape(x_val[:, :, :5], (-1, max_token_len, 5))
        x_token = decode_one_hot(x_token)
        x_token = tokenizer.batch_encode_plus(x_token, return_tensors="tf",
                                              padding="max_length", max_length=max_token_len // 6 + 1)
        x_token['input_ids'] = x_token['input_ids'].numpy()
        x_token['input_ids'] = x_token['input_ids'].reshape(
            x_val.shape[0], -1,
            x_token['input_ids'].shape[1])
        x_token['attention_mask'] = x_token['attention_mask'].numpy()
        x_token['attention_mask'] = x_token['attention_mask'].reshape(
            x_val.shape[0], -1,
            x_token['attention_mask'].shape[1])

        x = [[np.expand_dims(x_val[i], 0), x_token['input_ids'][i], x_token['attention_mask'][i]] for i in
             range(x_val.shape[0])]
        return x, y_val
    return (x_val, y_val)


def main_eval_model_tfrecord(args):
    print(f"args: {args}")
    val_data_path = args.val_data_path
    os.path.exists(val_data_path)
    val_data = load_t2t_data_tfrecord(
        dest_path=args.val_data_path,
        batch_size=args.batch_size,
        dataset_name="homo_sapiens_tiberius_20K",
        split="cal"
    )
    custom_objects = {}
    f1_factor = 2
    if f1_factor:
        cce_loss = custom_cce_f1_loss(2, batch_size=args.batch_size)
        custom_objects['custom_cce_f1_loss'] = cce_loss
        custom_objects['loss_'] = cce_loss
    else:
        cce_loss = tf.keras.losses.CategoricalCrossentropy()
    model = tf.keras.models.load_model(
        args.model,
        custom_objects=custom_objects
    )
    # result = model.evaluate(x=val_data[0], y=val_data[1], batch_size=args.batch_size, verbose=1)
    # print(';'.join(model.metrics_names), '\n', ';'.join(list(map(str, result))))
    eval_model(model, val_data)


def main_eval_model_pkl(args):
    print(args)
    sys.path.insert(0, args.learnMSA)
    # val_data_path = f'/home/gabriell/deepl_data/tfrecords/data/99999_hmm/val/validation_lstm.npz'
    val_data_path = args.val_data_path
    os.path.exists(val_data_path)
    assert args.max_length % 9 == 0, f"{args.max_length} //9 != 0"
    val_data = load_t2t_data_pkl(
        dest_path=args.val_data_path,
        batch_size=args.batch_size,
        dataset_name=args.val_data_name,
        split="val",
        max_length=args.max_length
    )

    custom_objects = {}
    f1_factor = 2
    if f1_factor:
        cce_loss = custom_cce_f1_loss(2, batch_size=args.batch_size)
        custom_objects['custom_cce_f1_loss'] = cce_loss
        custom_objects['loss_'] = cce_loss
    else:
        cce_loss = tf.keras.losses.CategoricalCrossentropy()
    model = tf.keras.models.load_model(
        args.model,
        custom_objects=custom_objects
    )
    eval_model(model, val_data, save_path=f"{args.save_path}", output_size=7, save_prefix=args.val_data_name)


def main():
    args = parseCmd()
    print(args)
    sys.path.insert(0, args.learnMSA)
    # val_data_path = f'/home/gabriell/deepl_data/tfrecords/data/99999_hmm/val/validation_lstm.npz'
    val_data_path = args.val_data_path
    assert os.path.exists(val_data_path), f"{val_data_path} does not exist"
    assert os.path.exists(args.model), f"{args.model} does not exist"
    assert os.path.exists(args.save_path), f"{args.save_path} does not exist"

    val_data = load_t2t_data_pkl(
        dest_path=args.val_data_path,
        batch_size=1,
        dataset_name="homo_sapiens_tiberius_20K",
        split="train"
    )

    custom_objects = {}
    f1_factor = 2
    if f1_factor:
        cce_loss = custom_cce_f1_loss(2, batch_size=args.batch_size)
        custom_objects['custom_cce_f1_loss'] = cce_loss
        custom_objects['loss_'] = cce_loss
    else:
        cce_loss = tf.keras.losses.CategoricalCrossentropy()
    model = tf.keras.models.load_model(
        args.model,
        custom_objects=custom_objects
    )
    eval_model(model, val_data)


def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(description='')
    #     parser.add_argument('--species', type=str,
    #         help='')
    parser.add_argument('--model', required=True, type=str,
                        help='')
    parser.add_argument('--val_data_path', type=str, required=True, default='.',
                        help='')
    parser.add_argument('--val_data_name', type=str, required=True, default='.',
                        help='')
    parser.add_argument('--save_path', type=str, required=True, default='.',
                        help='save path')
    parser.add_argument('--batch_size', type=int,
                        help='batch size')
    parser.add_argument('--max_length', type=int, default=9999,
                        help='seq length')
    parser.add_argument('--learnMSA', type=str, default='.',
                        help='')

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.random.set_random_seed(seed)
    tf.compat.v2.random.set_seed(seed)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parseCmd()
    set_seed(42)
    sys.path.insert(0, args.learnMSA)
    main_eval_model_pkl(args)
