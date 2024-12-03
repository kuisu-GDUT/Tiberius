export CUDA_VISIBLE_DEVICES=1
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius
#DATA_ROOT=/data/sukui_data/01_data/01_genomics_data/gene_structure

cd $TIBERIUS_ROOT
python bin/train_in_multispecies.py \
  --data /home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic \
  --dataset_name homo_sapiens_tiberius_20K \
  --train_species_file test \
  --val_data val \
  --out ./outputs/gorilla_15class_20k \
  --learnMSA .