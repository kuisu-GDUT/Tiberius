import sys, os

sys.path.append("./bin")
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from BCBio import GFF
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from eval_model_class import PredictionGTF
from models import lstm_model, custom_cce_f1_loss, add_hmm_layer

batch_size = 2
seq_len = 9999
strand = '+'

emb = False
hmm_factor = 1
trans_lstm = False

# inp_data_dir = 'test_data/Panthera_pardus/inp/'
# out_dir = 'outputs/test_train/'
inp_data_dir = 'test_data/annot_longest_fixed_paper'
out_dir = 'outputs/test_train_Human'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# genome_path = f'{inp_data_dir}/genome.fa'
# annot_path = f'{inp_data_dir}/annot.gtf'
genome_path = f'{inp_data_dir}/Homo_sapiens_GCF_000001405.40_GRCh38.p14_genomic.fna'
annot_path = f'{inp_data_dir}/Homo_sapiens.gtf'

pred_gtf = PredictionGTF(
    seq_len=seq_len,
    batch_size=batch_size,
    hmm=True,
    emb=False,
    num_hmm=1,
    hmm_factor=1,
    genome_path=genome_path,
    annot_path=annot_path,
    softmask=True,
    strand=strand,
    temp_dir=os.path.join(out_dir, "tmp")
)


class EpochSave(tf.keras.callbacks.Callback):
    def __init__(self, model_save_dir):
        super(EpochSave, self).__init__()
        self.model_save_dir = model_save_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f"{self.model_save_dir}/epoch_{epoch:02d}", save_traces=False)


# load input data x_seq
x_seq, y_seq, _ = pred_gtf.load_inp_data(
    strand=strand,
    chunk_coords=True,
    softmask=True,
    pad=False
)
print(f"x_seq shape: {x_seq.shape}, y_seq shape: {y_seq.shape}")

config = {
    "num_epochs": 10,
    "stride": 0,
    "units": 100,
    "filter_size": 32,
    "numb_lstm": 2,
    "numb_conv": 3,
    "dropout_rate": 0.0,
    "pool_size": 9,
    "lr": 1e-4,
    "batch_size": batch_size,
    "w_size": seq_len,
    'output_size': 15,
    'hmm_loss_weight_mul': 0.1,
    'hmm_emit_embeddings': False,
    "hmm_dense": 32,  # size of embedding for HMM input
    'hmm_share_intron_parameters': False,
    'hmm_nucleotides_at_exons': False,
    'hmm_trainable_transitions': False,
    'hmm_trainable_starting_distribution': False,
    'hmm_trainable_emissions': False,  # does not affect embedding emissions, use hmm_emit_embeddings for that
    "neutral_hmm": False,  # initializes an HMM without human expert bias, currently not implemented
    'hmm_factor': 99,
    'initial_variance': 0.1,
    'temperature': 32 * 3,
    "loss_f1_factor": 2.0,
}

relevant_keys = [
    'units', 'filter_size', 'kernel_size',
    'numb_conv', 'numb_lstm', 'dropout_rate',
    'pool_size', 'stride',
    'output_size', 'multi_loss'
]

relevant_args = {key: config[key] for key in relevant_keys if key in config}
model = lstm_model(**relevant_args)
add_hmm_layer(
    model,
    gene_pred_layer=None,
    dense_size=config['hmm_dense'],
    pool_size=config['pool_size'],
    output_size=config['output_size'],
    num_hmm=1,
    l2_lambda=0.,
    hmm_factor=config['hmm_factor'],
    batch_size=config['batch_size'],
    seq_len=config['w_size'],
    initial_variance=config['initial_variance'],
    temperature=config['temperature'],
    emit_embeddings=False,
    share_intron_parameters=config['hmm_share_intron_parameters'],
    trainable_nucleotides_at_exons=config['hmm_nucleotides_at_exons'],
    trainable_emissions=config['hmm_trainable_emissions'],
    trainable_transitions=config['hmm_trainable_transitions'],
    trainable_starting_distribution=config['hmm_trainable_starting_distribution'],
    use_border_hints=False,
    include_lstm_in_output=False,
    neutral_hmm=config['neutral_hmm']
)

adam = Adam(learning_rate=config["lr"])
f1loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])

model.compile(loss=f1loss, optimizer=adam, metrics=['accuracy'])
model.summary()

model_save = out_dir + "/weights.{epoch:02d}"
epoch_callback = EpochSave(out_dir)
model.fit(
    x=x_seq,
    y=y_seq,
    epochs=config["num_epochs"],
    batch_size=config["batch_size"],
    callbacks=[epoch_callback]
)
