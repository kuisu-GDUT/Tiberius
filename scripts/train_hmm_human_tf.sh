export CUDA_VISIBLE_DEVICES=2
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius
#DATA_ROOT=/data/sukui_data/01_data/01_genomics_data/gene_structure

cd $TIBERIUS_ROOT
python bin/train_in_human.py \
  --data /home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/Tiberius/tfrecords/Homo_sapiens \
  --train_species_file train_species_filtered.txt \
  --val_data val_species_filtered.txt \
  --out ./outputs/Human_15class_tfrecord \
  --learnMSA . \
  --load weights/tiberius_weights