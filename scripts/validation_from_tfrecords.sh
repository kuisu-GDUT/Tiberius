export CUDA_VISIBLE_DEVICES=3
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius
#DATA_ROOT=/data/sukui_data/01_data/01_genomics_data/gene_structure

cd $TIBERIUS_ROOT

python bin/validation_from_tfrecords.py \
  --tfrec_dir ~/02_data/07_genomics_data/Tiberius/tfrecords/Homo_sapiens/ \
  --species val_species_filtered.txt \
  --batch_size 1 \
  --val_size 2 \
  --output_size 7 \
  --out outputs/Human_15class_tfrecord/validation_data_10