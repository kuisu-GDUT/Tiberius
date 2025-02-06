import glob
import logging
import pickle
import random
import sys, os, json, csv, argparse
import logging

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

logging.basicConfig(level=logging.INFO)

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


def tokenizer(sequence):
    table = np.zeros((256, 6), dtype=np.uint8)
    table[:, 4] = 1  # N is encoded as [0, 0, 0, 0, 1, 0]

    # Set specific labels for A, C, G, T
    table[ord('A'), :] = [1, 0, 0, 0, 0, 0]
    table[ord('C'), :] = [0, 1, 0, 0, 0, 0]
    table[ord('G'), :] = [0, 0, 1, 0, 0, 0]
    table[ord('T'), :] = [0, 0, 0, 1, 0, 0]
    # Set labels for a, c, g, t with softmasking indicator
    table[ord('a'), :] = [1, 0, 0, 0, 0, 1]
    table[ord('c'), :] = [0, 1, 0, 0, 0, 1]
    table[ord('g'), :] = [0, 0, 1, 0, 0, 1]
    table[ord('t'), :] = [0, 0, 0, 1, 0, 1]

    # Convert the sequence to integer sequence
    int_seq = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
    # Perform one-hot encoding
    return table[int_seq]


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


def load_t2t_data_chr_txt(fasta_root, label_root, csv_path, max_length=9999, save_path=None):
    # load bend data
    assert os.path.exists(csv_path), f"csv file not found: {csv_path}"
    assert os.path.exists(fasta_root), f"fasta root not found: {fasta_root}"
    assert os.path.exists(label_root), f"label root not found: {label_root}"

    df_csv = pd.read_csv(csv_path, header=0)

    # get strand
    strand = []
    for index, row in df_csv.iterrows():
        path_name = row[0]
        if "forward" in path_name:
            strand.append("forward")
        else:
            strand.append("backward")

    df_csv["strand"] = strand

    df_cds_csv = df_csv[(df_csv['y_pred'] == 1) | (df_csv['y_true'] == 1)]
    df_cds_csv = df_cds_csv.sort_values(by=["chrom", "strand", "seq_start"])
    chr_names = set(df_cds_csv["chrom"].unique())
    strands = set(df_cds_csv["strand"].unique())
    for chr in chr_names:
        for strand in strands:
            file_part_name = chr.split("NC")[-1]
            chr_strand_name = f"*{file_part_name}*_{strand}*"
            fasta_path = glob.glob(os.path.join(fasta_root, chr_strand_name))[0]
            with open(fasta_path, 'r') as f:
                fasta_seq = f.read().strip()
            label_path = glob.glob(os.path.join(label_root, chr_strand_name))[0]
            logging.info(fasta_path)
            logging.info(label_path)
            with open(label_path, 'rb') as f:
                label_data = pickle.load(f)
                label_data = label_data.toarray()

            assert len(fasta_seq) == len(label_data)

            seqs = []
            labels = []
            chunk_infos = []
            for index, row in df_cds_csv[(df_cds_csv['chrom'] == chr) & (df_cds_csv['strand'] == strand)].iterrows():
                start = row["seq_start"]
                end = row['seq_end']
                input_seq = fasta_seq[start:end]
                input_label = label_data[start:end]
                seqs.append(input_seq)
                labels.append(input_label)
                chunk_infos.append([chr, strand, start, end, row["y_pred"]])

            # start inference
            input_seq = ""
            input_label = None
            input_chunk = []
            i = 0
            for idx, (seq, label, chunk_info) in enumerate(zip(seqs, labels, chunk_infos)):
                # predict None CDS
                if chunk_info[-1] == 0:
                    chr_start = chunk_info[2]
                    chr_end = chunk_info[3]
                    seq_predict = chunk_info[4]
                    name = f"{chr}_{strand}_{chr_start}_{chr_end}_{seq_predict}.pkl"
                    if len(label.shape) == 2:
                        output_label = np.argmax(label, axis=-1)
                    else:
                        output_label = label

                    y_pred_tmp = np.zeros_like(output_label)
                    with open(os.path.join(save_path, name), 'wb') as f:
                        pred_result = {"chr": chr, "strand": strand, "start": chr_start, "end": chr_end,
                                       "y_pred": y_pred_tmp, "y_true": output_label}
                        pickle.dump(pred_result, f)
                    continue

                input_seq += seq
                if input_label is None:
                    input_label = label
                else:
                    input_label = np.concatenate([input_label, label], axis=0)
                chunk_info += [len(seq) * i, len(seq) * (i + 1)]
                logging.info(chunk_info)
                input_chunk.append(chunk_info)
                i += 1

                if len(input_seq) > max_length:
                    _input_seq = tokenizer(input_seq)
                    yield np.expand_dims(_input_seq, axis=0), np.expand_dims(input_label, axis=0), input_chunk
                    input_seq = ""
                    input_label = None
                    input_chunk = []
                    i = 0


def eval_model(model, val_data, save_path=None, output_size: int = 15):
    logging.info("Evaluating model on validation data")
    labels = []
    features = []
    y_predicts = []
    for i, val_i_data in tqdm.tqdm(enumerate(val_data), desc="evaluating"):
        input_seq, input_label, input_chunk = val_i_data
        mod_value = input_seq.shape[1] % 9
        if mod_value != 0:
            feature = input_seq[:, :input_seq.shape[1] - mod_value]
            label = input_label[:, :input_seq.shape[1] - mod_value]
        else:
            feature = input_seq
            label = input_label

        y_predict = model.predict(feature)
        # get class of predict
        y_predict_label = np.argmax(y_predict, axis=-1)
        y_predict_onehot = np.eye(y_predict.shape[-1])[y_predict_label]
        y_predicts.append(y_predict_onehot)

        # save predict result
        for chunk_info in input_chunk:
            try:
                chr, strand, chr_start, chr_end, seq_predict, chunk_start, chunk_end = chunk_info
                # get predict result
                y_pred = y_predict_onehot[:, chunk_start:chunk_end]
                logging.info(y_pred.shape)
                y_pred = np.argmax(y_pred, axis=-1)
                if chunk_end - chunk_start != y_pred.shape[1]:
                    error_len = chunk_end - chunk_start - y_pred.shape[1]
                    y_pred = np.concatenate([y_pred, np.zeros_like(y_pred)[:, :error_len]], axis=1)

                # get label result
                y_true = label[:, chunk_start:chunk_end]
                logging.info(y_true.shape)
                y_true = np.argmax(y_true, axis=-1)
                if chunk_end-chunk_start != y_true.shape[1]:
                    error_len = chunk_end-chunk_start - y_true.shape[1]
                    y_true = np.concatenate([y_true, np.zeros_like(y_true)[:, :error_len]], axis=1)

                name = f"{chr}_{strand}_{chr_start}_{chr_end}_{seq_predict}.pkl"
                with open(os.path.join(save_path, name), 'wb') as f:
                    pred_result = {"chr": chr, "strand": strand, "start": chr_start, "end": chr_end,
                                   "y_pred": y_pred[0],
                                   "y_true": y_true[0]}
                    pickle.dump(pred_result, f)

            except Exception as e:
                logging.info(f"error: {e}")
                logging.info(f"chunk_info: {chunk_info}")

        labels.append(label)
        features.append(feature)
        onehot_label = np.argmax(label, axis=-1)
        if i % 10 == 0:
            logging.info(
                f"i:{i}; input_seq shape: {input_seq.shape}, input_label shape: {input_label.shape}, mod_value: {mod_value}")
            logging.info(feature.shape)
            logging.info(
                f"i:{i}; predict shape: {y_predict.shape}, feature shape: {feature.shape}, label shape: {label.shape}  "
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
    if y_predicts.shape[-1] > output_size:
        logging.info(f"reduce predicts from {y_predicts.shape[-1]} to {output_size}")
        y_predicts = tiberius_reduce_labels(y_predicts, output_size)
    if labels.shape[-1] > output_size:
        logging.info(f"reduce labels from {labels.shape[-1]} to {output_size}")
        labels = tiberius_reduce_labels(labels, output_size)

    logging.info(f"label distribution each class: {y_df.value_counts()}")
    logging.info(f"predicts shape: {y_predicts.shape}; label shape: {labels.shape}; catagories: {np.unique(y_catagories)}")
    logging.info(f"predicts shape: {y_predicts.shape}; label shape: {labels.shape}")

    cal_metric(labels, y_predicts)


def main_eval_model_txt(args):
    logging.info(args)
    sys.path.insert(0, args.learnMSA)
    # val_data_path = f'/home/gabriell/deepl_data/tfrecords/data/99999_hmm/val/validation_lstm.npz'
    assert args.max_length % 9 == 0, f"{args.max_length} //9 != 0"

    val_data = load_t2t_data_chr_txt(
        fasta_root=args.fasta_root,
        label_root=args.label_root,
        csv_path=args.csv_path,
        max_length=args.max_length,
        save_path=args.save_path,
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
    eval_model(model, val_data, save_path=args.save_path, output_size=7)


def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(description='')
    #     parser.add_argument('--species', type=str,
    #         help='')
    parser.add_argument('--model', required=False, type=str, default='.',
                        help='')
    parser.add_argument('--fasta_root', type=str, required=True, default='.', help='')
    parser.add_argument('--label_root', type=str, required=True, default='.', help='')
    parser.add_argument('--csv_path', type=str, required=True, default='.', help='')
    parser.add_argument('--save_path', type=str, required=False, default='.',
                        help='save path')
    parser.add_argument('--batch_size', type=int, default=2,
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
    main_eval_model_txt(args)
    # for i in load_t2t_data_chr_txt(max_length=2997):
    #     input_seq, input_label, input_chunk = i
    #     logging.info(input_seq.shape, input_label.shape, input_chunk[:3])
