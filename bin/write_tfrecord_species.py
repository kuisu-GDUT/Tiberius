import pickle
import sys, json, os, re, sys, csv, argparse
from typing import Optional

import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from transformers import AutoTokenizer, TFAutoModelForMaskedLM, TFEsmForMaskedLM
from genome_fasta import GenomeSequences
from annotation_gtf import GeneStructure
import subprocess as sp
import numpy as np
import tensorflow as tf
import numpy as np
# import psutil
import sys
import zlib
from copy import deepcopy
from wig_class import Wig_util

from concurrent.futures import ThreadPoolExecutor

import h5py


# def get_species_data_lstm(s, seq_len=500004, overlap_size=0):
#     genome_path = f'/home/gabriell/deepl_data/genomes/{s}.fa.combined.masked'
#     annot_path=f'/home/gabriell//deepl_data/annot_longest_fixed/{s}.gtf'

#     fasta = GenomeSequences(fasta_file=genome_path,
#             chunksize=seq_len, 
#             overlap=overlap_size)
#     fasta.encode_sequences() 
#     f_chunk = fasta.get_flat_chunks(strand='+')

#     full_f_chunks = np.concatenate((f_chunk[::-1,::-1, [3,2,1,0,4,5]], 
#                                     f_chunk), axis=0)

#     seq_len = [len(s) for s in fasta.sequences]
#     del f_chunk
#     del fasta.sequences
#     del fasta.one_hot_encoded

#     ref_anno = GeneStructure(annot_path, 
#                         chunksize=seq_len, 
#                         overlap=overlap_size)

#     ref_anno.translate_to_one_hot(fasta.sequence_names, 
#                             seq_len)
#     #ref_anno.get_flat_chunks(fasta.sequence_names)

#     r_chunk, r_phase = ref_anno.get_flat_chunks(fasta.sequence_names, strand='-')
#     r_chunk2, r_phase2 = ref_anno.get_flat_chunks(fasta.sequence_names, strand='+')

#     full_r_chunks = np.concatenate((r_chunk, r_chunk2), axis=0)
#     full_r_phase_chunks = np.concatenate((r_phase, r_phase2), axis=0)

#     return full_f_chunks, full_r_chunks, full_r_phase_chunks

def get_transformer_emb(chunk, token_len=5994):
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    transformer_model = TFEsmForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")

    def decode_one_hot(encoded_seq):
        index_to_nucleotide = np.array(['A', 'C', 'G', 'T', 'A'])
        nucleotide_indices = np.argmax(encoded_seq, axis=-1)
        decoded_seq = index_to_nucleotide[nucleotide_indices]
        decoded_seq_str = [''.join(seq) for seq in decoded_seq]
        return decoded_seq_str

    inp_chunk = decode_one_hot(chunk.reshape(-1, token_len, 6)[:, :, :5])
    tokens = tokenizer.batch_encode_plus(inp_chunk, return_tensors="tf",
                                         padding="max_length",
                                         max_length=token_len // 6 + 1)

    emb = transformer_model(tokens['input_ids'],
                            attention_mask=tokens['attention_mask'],
                            output_hidden_states=True)
    trans_out = emb['hidden_states'][-1][:, 1:].numpy()
    return np.array(trans_out)


def get_clamsa_track(file_path, seq_len=500004, prefix=''):
    wig = Wig_util()
    seq = []
    with open(f'{file_path}/../{prefix}_seq_names.txt', 'r') as f:
        for line in f.readlines():
            seq.append(line.strip().split())
    for s1, s2 in zip(['+', '-'], ['plus', 'minus']):
        for phase in [0, 1, 2]:
            print(f'{file_path}/{prefix}_{phase}-{s2}.wig', file=sys.stderr)
            wig.addWig2numpy(f'{file_path}/{prefix}_{phase}-{s2}.wig', seq, strand=s1)
    chunks_plus = wig.get_chunks(chunk_len=seq_len, sequence_names=[s[0] for s in seq])
    return np.concatenate([chunks_plus[::-1, ::-1, [1, 0, 3, 2]], chunks_plus], axis=0)


def load_clamsa_data(clamsa_prefix, seq_names, seq_len=None):
    clamsa_chunks = []
    seq = []
    with open(seq_names, 'r') as f:
        for line in f.readlines():
            seq.append(line.strip().split())
    for s in seq:
        if not os.path.exists(f'{clamsa_prefix}{s}.npy'):
            print(f'CLAMSA PATH {clamsa_prefix}{s}.npy does not exist!')
        clamsa_array = np.load(f'{clamsa_prefix}{s}.npy')
        numb_chunks = clamsa_array.shape[0] // seq_len
        clamsa_array_new = clamsa_array[:numb_chunks * seq_len].reshape(numb_chunks, seq_len, 4)
        clamsa_chunks.append(clamsa_array_new)

    clamsa_chunks = np.concatenate(clamsa_chunks, axis=0)
    return np.concatenate([clamsa_chunks[::-1, ::-1, [1, 0, 3, 2]], clamsa_chunks], axis=0)


def get_species_data_hmm(genome_path='', annot_path='', species='', seq_len=500004, overlap_size=0, transition=False):
    if not genome_path:
        genome_path = f'/home/gabriell/deepl_data/genomes/{species}.fa.combined.masked'
    if not annot_path:
        annot_path = f'/home/gabriell//deepl_data/annot_longest_fixed/{species}.gtf'

    fasta = GenomeSequences(
        fasta_file=genome_path,
        chunksize=seq_len,
        overlap=overlap_size
    )
    fasta.encode_sequences()
    seqs = [len(s) for s in fasta.sequences]
    seq_names = fasta.sequence_names
    f_chunk = fasta.get_flat_chunks(strand='+', pad=False)
    del fasta
    print(f_chunk.shape)
    full_f_chunks = np.concatenate((f_chunk[::-1, ::-1, [3, 2, 1, 0, 4, 5]],
                                    f_chunk), axis=0)

    del f_chunk
    # del fasta
    print(full_f_chunks.shape)
    ref_anno = GeneStructure(annot_path,
                             chunksize=seq_len,
                             overlap=overlap_size)

    ref_anno.translate_to_one_hot_hmm(seq_names,
                                      seqs, transition=transition)
    del ref_anno.gene_structures

    full_r_chunks = np.concatenate((ref_anno.get_flat_chunks_hmm(seq_names, strand='-'),
                                    ref_anno.get_flat_chunks_hmm(seq_names, strand='+')),
                                   axis=0)
    del ref_anno

    return full_f_chunks, full_r_chunks


def write_h5(fasta, ref, out, ref_phase=None, split=100,
             trans=False, clamsa=np.array([])):
    fasta = fasta.astype(np.int32)
    ref = ref.astype(np.int32)

    file_size = fasta.shape[0] // split
    indices = np.arange(fasta.shape[0])
    np.random.shuffle(indices)
    print(clamsa.shape, fasta.shape, trans)
    if ref_phase:
        ref_phase = ref_phase.astype(np.int32)
    for k in range(split):
        # Create a new HDF5 file with compression
        with h5py.File(f'{out}_{k}.h5', 'w') as f:
            # Create datasets with GZIP compression
            f.create_dataset('input', data=fasta[indices[k::split]], compression='gzip',
                             compression_opts=9)  # Maximum compression
            f.create_dataset('output', data=ref[indices[k::split]], compression='gzip', compression_opts=9)


def write_numpy(fasta, ref, out, ref_phase=None, split=100, trans=False, clamsa=np.array([])):
    fasta = fasta.astype(np.int32)
    ref = ref.astype(np.int32)

    file_size = fasta.shape[0] // split
    indices = np.arange(fasta.shape[0])
    np.random.shuffle(indices)
    print(clamsa.shape, fasta.shape, trans)
    if ref_phase:
        ref_phase = ref_phase.astype(np.int32)
    for k in range(split):
        print(f'Writing numpy split {k + 1}/{split}')
        if clamsa.any():
            np.savez(f'{out}_{k}.npz', array1=fasta[indices[k::split], :, :],
                     array2=ref[indices[k::split], :, :], array3=clamsa[indices[k::split], :, :])
        else:
            np.savez(f'{out}_{k}.npz', array1=fasta[indices[k::split], :, :],
                     array2=ref[indices[k::split], :, :], )


def write_pkl(fasta, ref, out, split=1, ref_phase=None, trans=False, clamsa=np.array([])):
    fasta = fasta.astype(np.int32)
    ref = ref.astype(np.int32)
    seq_len = fasta.shape[1]

    print(clamsa.shape, fasta.shape, trans)
    for idx, (seq, label) in tqdm.tqdm(enumerate(zip(fasta, ref)), desc='Writing pkl files', total=len(fasta)):
        start_idx = idx * seq_len
        data = {
            "seq": seq,
            "annotation": label,
            "start_idx": start_idx,
            "end_idx": start_idx + seq_len,
        }
        with open(f'{out}_{start_idx}-{start_idx+seq_len}.pkl', 'wb') as f:
            pickle.dump(data, f)

def write_tf_record(fasta, ref, out, ref_phase=None, split=100, trans=False, clamsa=np.array([])):

    indices = np.arange(fasta.shape[0])
    np.random.shuffle(indices)
    print(f"clamsa shape: {clamsa.shape}' fasta  shape: {fasta.shape}; ref shape: {ref.shape}; trans: {trans}")
    if ref_phase:
        ref_phase = ref_phase.astype(np.int32)

    def create_example(i):
        fasta_i = fasta[i, :, :].astype(np.int32)
        ref_i = ref[i, :, :].astype(np.int32)

        feature_bytes_x = tf.io.serialize_tensor(fasta_i).numpy()
        feature_bytes_y = tf.io.serialize_tensor(ref_i).numpy()
        if ref_phase is not None:
            ref_phase_i = ref_phase[i, :, :].astype(np.int32)
            feature_bytes_y_phase = tf.io.serialize_tensor(ref_phase_i).numpy()
            example = tf.train.Example(features=tf.train.Features(feature={
                'input': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                'output': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_y])),
                'output_phase': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_y_phase]))
            }))
        elif trans:
            trans_emb = get_transformer_emb(ref_i, token_len=fasta_i.shape[1] // 18)
            feature_bytes_trans = tf.io.serialize_tensor(trans_emb).numpy()
            example = tf.train.Example(features=tf.train.Features(feature={
                'input': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                'output': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_y])),
                'trans_emb': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_trans]))
            }))
        elif clamsa.any():
            feature_bytes_clamsa = tf.io.serialize_tensor(clamsa[i, :, :]).numpy()
            example = tf.train.Example(features=tf.train.Features(feature={
                'input': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                'output': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_y])),
                'clamsa': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_clamsa]))
            }))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'input': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                'output': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_y]))
            }))
        return example.SerializeToString()

    for k in range(split):
        print(f'Writing split {k + 1}/{split}')
        with tf.io.TFRecordWriter(f'{out}_{k}.tfrecords',
                                  options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
            for i in indices[k::split]:
                serialized_example = create_example(i)
                writer.write(serialized_example)

    print('Writing complete.')


"""
def write_tf_record(fasta, ref, out, ref_phase=None, split=100, 
                    trans=False, clamsa=np.array([])):
    fasta = fasta.astype(np.int32)          
    ref = ref.astype(np.int32)

    file_size = fasta.shape[0] // split
    indices = np.arange(fasta.shape[0])
    np.random.shuffle(indices)
    print(clamsa.shape, fasta.shape, trans)
    if ref_phase:
        ref_phase = ref_phase.astype(np.int32)
    for k in range(split):
        print('k')
        with tf.io.TFRecordWriter(f'{out}_{k}.tfrecords' , \
            options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer: 
            for i in indices[k::split]:
                feature_bytes_x = tf.io.serialize_tensor(fasta[i,:,:]).numpy()
                feature_bytes_y = tf.io.serialize_tensor(ref[i,:,:]).numpy()
                if ref_phase:
                    feature_bytes_y_phase = tf.io.serialize_tensor(ref_phase[i,:,:]).numpy()                
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'input': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                        'output': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_y])),
                        'output_phase': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_y_phase]))
                    }))
                elif trans:
                    trans_emb = get_transformer_emb(ref[i,:,:], token_len = fasta.shape[1]//18)
                    print('VV', file=sys.stderr)
                    feature_bytes_trans = tf.io.serialize_tensor(trans_emb).numpy()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'input': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                        'output': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_y])),
                        'trans_emb': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_trans]))
                    }))
                elif clamsa.any():
                    feature_bytes_clamsa = tf.io.serialize_tensor(clamsa[i,:,:]).numpy()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'input': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                        'output': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_y])),
                        'clamsa': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_clamsa]))
                    }))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'input': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                        'output': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_y]))
                    }))
                serialized_example = example.SerializeToString()
                writer.write(serialized_example)"""


def write_species_data_hmm(
        genome_path='',
        annot_path='',
        species='',
        seq_len=500004,
        overlap_size=0,
        transition=True,
        out_name='',
        split=10,
        args: Optional[dict] = None
):
    if not genome_path:
        genome_path = f'/home/gabriell/deepl_data/genomes/{species}.fa.combined.masked'
    if not annot_path:
        annot_path = f'/home/gabriell/deepl_data/annot_longest_fixed/{species}.gtf'

    fasta = GenomeSequences(
        fasta_file=genome_path,
        chunksize=seq_len,
        overlap=overlap_size
    )
    ref_anno = GeneStructure(
        annot_path,
        chunksize=seq_len,
        overlap=overlap_size
    )
    for seq_name in tqdm.tqdm(fasta.sequence_names, desc="Writing species data"):
        seq_len = len(fasta.sequences[fasta.sequence_names.index(seq_name)])
        print(f"process seq: {seq_name}, len:{seq_len}")
        seq_lens = [seq_len]
        seq_names = [seq_name]
        out_seq_name = f"{out_name}_{seq_name}_-+"

        fasta.encode_sequences(seq=[seq_name])
        # seqs = [len(s) for s in fasta.sequences]
        # seq_names = fasta.sequence_names
        f_chunk = fasta.get_flat_chunks(strand='+', sequence_name=seq_names, pad=False)
        print(f_chunk.shape)
        full_f_chunks = np.concatenate(
            (f_chunk[::-1, ::-1, [3, 2, 1, 0, 4, 5]], f_chunk),
            axis=0)

        del f_chunk
        # del fasta
        print(full_f_chunks.shape)

        ref_anno.translate_to_one_hot_hmm(
            seq_names,
            seq_lens,
            transition=transition)

        full_r_chunks = np.concatenate(
            (ref_anno.get_flat_chunks_hmm(seq_names, strand='-'), ref_anno.get_flat_chunks_hmm(seq_names, strand='+')),
            axis=0)

        if args.transformer:
            write_tf_record(full_f_chunks, full_r_chunks, out_seq_name, trans=True, split=split)
        if args.clamsa:
            # clamsa = get_clamsa_track('/home/gabriell/deepl_data/clamsa/wig/', seq_len=args.wsize, prefix=args.species)
            clamsa = load_clamsa_data(args.clamsa, seq_names=args.seq_names, seq_len=args.wsize)
            print('Loaded CLAMSA')
            if args.np:
                write_numpy(full_f_chunks, full_r_chunks, out_seq_name, clamsa=clamsa, split=split)
            else:
                write_tf_record(full_f_chunks, full_r_chunks, out_seq_name, clamsa=clamsa, split=split)
        else:
            if args.h5:
                write_h5(full_f_chunks, full_r_chunks, out_seq_name, split=split)
            elif args.np:
                write_numpy(full_f_chunks, full_r_chunks, out_seq_name, split=split)
            elif args.pkl:
                write_pkl(full_f_chunks, full_r_chunks, out_seq_name, split=split)
            else:
                write_tf_record(full_f_chunks, full_r_chunks, out_seq_name, split=split)


def main():
    args = parseCmd()

    write_species_data_hmm(
        genome_path=args.fasta,
        annot_path=args.gtf,
        species=args.species,
        seq_len=args.wsize,
        overlap_size=0,
        transition=args.transition,
        out_name=args.out,
        args=args
    )  # NOTE: defalut transition=True

    # print('Loaded FASTA and GTF', fasta.shape, ref.shape)
    # if args.transformer:
    #     #         trans_emb = get_transformer_emb(ref, token_len = args.wsize//18)
    #     #         print('AAA')
    #     write_tf_record(fasta, ref, args.out, trans=True)
    # if args.clamsa:
    #     # clamsa = get_clamsa_track('/home/gabriell/deepl_data/clamsa/wig/', seq_len=args.wsize, prefix=args.species)
    #     clamsa = load_clamsa_data(args.clamsa, seq_names=args.seq_names, seq_len=args.wsize)
    #     print('Loaded CLAMSA')
    #     if args.np:
    #         write_numpy(fasta, ref, args.out, clamsa=clamsa)
    #     else:
    #         write_tf_record(fasta, ref, args.out, clamsa=clamsa)
    # else:
    #     if args.h5:
    #         write_h5(fasta, ref, args.out)
    #     elif args.np:
    #         write_numpy(fasta, ref, args.out)
    #     else:
    #         write_tf_record(fasta, ref, args.out)


def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(description="""
    USAGE: write_tfrecord_species.py --gtf annot.gtf --fasta genome.fa --wsize 9999 --out tfrecords/speciesName
    
    This script will write input and output data as 100 tfrecord files as tfrecords/speciesName_i.tfrecords""")
    parser.add_argument('--species', type=str, default='',
                        help='')
    parser.add_argument('--gtf', type=str, default='', required=True,
                        help='Annotation in GTF format.')
    parser.add_argument('--fasta', type=str, default='', required=True,
                        help='Genome sequence in FASTA format.')
    parser.add_argument('--out', type=str, required=True,
                        help='Prefix of output files')
    parser.add_argument('--wsize', type=int,
                        help='', required=True)
    parser.add_argument('--transition', action='store_true',
        help='')
    parser.add_argument('--transformer', action='store_true',
                        help='')
    parser.add_argument('--clamsa', type=str, default='',
                        help='')
    parser.add_argument('--seq_names', type=str, default='',
                        help='')
    parser.add_argument('--h5', action='store_true',
                        help='')
    parser.add_argument('--np', action='store_true',
                        help='')
    parser.add_argument('--pkl', action='store_true',
                        help='')

    return parser.parse_args()


if __name__ == '__main__':
    main()
