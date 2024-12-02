# ==============================================================
# Authors: Lars Gabriel
#
# Get validation data from tfrecords files. 
# Creates a numpy array from sampels from tfrecords files.
# ==============================================================
import glob
import os, sys, json, argparse
import numpy as np
import pandas as pd
import tqdm

from data_generator import DataGenerator
import tensorflow as tf


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


def main():
    args = parseCmd()
    # species = read_species(args.species)
    # file_paths = [f'{args.tfrec_dir}/{s}_{i}.tfrecords' for s in species for i in range(args.tfrec_per_species)]

    species = read_species(f'{args.tfrec_dir}/{args.species}')
    file_paths = []
    for specie in species:
        file_paths.extend(glob.glob(f'{args.tfrec_dir}/{specie}_*.tfrecords'))
    data_x = []
    data_y = []
    data_clamsa = []

    generator = DataGenerator(
        file_path=file_paths,
        batch_size=args.batch_size,
        shuffle=False,
        repeat=True,
        # max_nums=100000,
        filter=False,
        output_size=args.output_size,
        hmm_factor=0,
        trans=False, clamsa=args.clamsa
    )

    for j in tqdm.tqdm(range(args.val_size), desc='reading data', total=args.val_size):
        i = np.random.randint(0, args.batch_size)
        if args.clamsa:
            example_x, example_y, example_clamsa = next(generator)
            data_clamsa.append(example_clamsa[i][2])
        else:
            example_x, example_y = next(generator)
        data_x.append(example_x[i])
        data_y.append(example_y[i])

    data_y = np.array(data_y)
    data_x = np.array(data_x)
    y_catagories = np.argmax(data_y, axis=-1)
    y_df = pd.DataFrame(y_catagories.flatten())
    print(f"label distribution each class: {y_df.value_counts()}")
    print(f"data_x shape: {data_x.shape}; data_y shape: {data_y.shape}; y_catagories: {np.unique(y_catagories)}")
    print(f"save in {args.out}")
    if args.clamsa:
        np.savez(args.out, array1=np.array(data_x), array2=np.array(data_y), array3=np.array(data_clamsa))
    else:
        np.savez(args.out, array1=np.array(data_x), array2=np.array(data_y))


def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(

        description="""Get validation data from tfrecords files. 
        Creates a numpy array from sampels from tfrecords files.
        The tfrecord files have to be named like <species>_<number>.tfrecords .
    """)
    parser.add_argument('--tfrec_dir', type=str,
                        help='Location of the tfrecords files.')
    parser.add_argument('--species', type=str,
                        help='Text file with species names. The tfrecords have to have the species name as prefix.')
    parser.add_argument('--tfrec_per_species', type=int, default=100,
                        help='Number of tfRecord files per species. Default 100.')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size used by loading tfrecords, lower this if it does not fit into memory.')
    parser.add_argument('--val_size', type=int, default=2000,
                        help='Number of trainings examples in the output.')
    parser.add_argument('--clamsa', action='store_true')
    parser.add_argument('--out', type=str,
                        help='Output name of numpy array.', default='validation_data')
    parser.add_argument('--output_size', type=str,
                        help='output size, including 7, 15', default=15)
    return parser.parse_args()


if __name__ == '__main__':
    main()
