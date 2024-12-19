import pickle
import sys, json, os, re, sys, csv, argparse
from typing import Optional

import tqdm
from scipy.sparse import csr_matrix

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
    full_f_chunks = np.concatenate((f_chunk,
                                    f_chunk[::-1, ::-1, [3, 2, 1, 0, 4, 5]]),
                                   axis=0)

    del f_chunk
    # del fasta
    print(full_f_chunks.shape)
    ref_anno = GeneStructure(annot_path,
                             chunksize=seq_len,
                             overlap=overlap_size)

    ref_anno.translate_to_one_hot_hmm(seq_names,
                                      seqs, transition=transition)
    del ref_anno.gene_structures

    full_r_chunks = np.concatenate((ref_anno.get_flat_chunks_hmm(seq_names, strand='+'),
                                    ref_anno.get_flat_chunks_hmm(seq_names, strand='-')),
                                   axis=0)
    del ref_anno

    return full_f_chunks, full_r_chunks


def encode_sequence(sequence):
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
    return table[int_seq]


def decode_sequence(encoded_seq):
    index_to_nucleotide = np.array(['A', 'C', 'G', 'T', 'N', 'a', 'c', 'g', 't'])
    nucleotide_indices = np.argmax(encoded_seq, axis=-1)
    if encoded_seq.shape[-1] == 6:
        soft_mask = encoded_seq[..., -1] == 1
        nucleotide_indices[soft_mask] += 5
    decoded_seq = index_to_nucleotide[nucleotide_indices]
    decoded_seq_str = ''.join(decoded_seq)
    check_flag = encoded_seq != encode_sequence(decoded_seq_str)
    if check_flag.any():
        print(f"decoded_seq_str: {decoded_seq_str}")
        print(f"encode seq: {encoded_seq}")
    # if encoded_seq != encode_sequence(decoded_seq_str):
    return decoded_seq_str


def write_pkl(fasta, ref, out, split=1, ref_phase=None, trans=False, clamsa=np.array([]), strand="+-"):
    seq_len = fasta.shape[1]

    print(f"clamsa.shape {clamsa.shape}, fasta shape: {fasta.shape}, ref shape: {ref.shape}, trans: {trans}")
    for idx, (seq, label) in tqdm.tqdm(enumerate(zip(fasta, ref)), desc='Writing pkl files', total=len(fasta)):
        start_idx = idx * seq_len  # NOTE, idex start 1 in CHR
        data = {
            # "input_id": csr_matrix(seq),
            "seq": decode_sequence(seq),
            "annotation": csr_matrix(label),
            "strand": strand,
            "start_idx": start_idx,
            "end_idx": start_idx + seq_len,
        }
        with open(f'{out}_{strand}_{start_idx}-{start_idx + seq_len}.pkl', 'wb') as f:
            pickle.dump(data, f)


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
        for strand in ['+', '-']:
            seq_len = len(fasta.sequences[fasta.sequence_names.index(seq_name)])
            print(f"process seq: {seq_name}, len:{seq_len}, srand: {strand}")
            seq_lens = [seq_len]
            seq_names = [seq_name]
            out_seq_name = f"{out_name}_{seq_name}"

            fasta.encode_sequences(seq=[seq_name])
            # seqs = [len(s) for s in fasta.sequences]
            # seq_names = fasta.sequence_names
            full_f_chunks = fasta.get_flat_chunks(strand=strand, sequence_name=seq_names, pad=False)

            ref_anno.translate_to_one_hot_hmm(
                seq_names,
                seq_lens,
                transition=transition)
            full_r_chunks = ref_anno.get_flat_chunks_hmm(seq_names, strand=strand)
            print(f"strand {strand}. fasta shape: {full_f_chunks.shape}. ref shape: {full_r_chunks.shape}", )

            if args.clamsa:
                # clamsa = get_clamsa_track('/home/gabriell/deepl_data/clamsa/wig/', seq_len=args.wsize, prefix=args.species)
                clamsa = load_clamsa_data(args.clamsa, seq_names=args.seq_names, seq_len=args.wsize)
                print('Loaded CLAMSA')
            else:
                if args.pkl:
                    write_pkl(full_f_chunks, full_r_chunks, out_seq_name, split=split, strand=strand)
                else:
                    raise ValueError("No output format specified")

            del full_f_chunks
            del full_r_chunks


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
