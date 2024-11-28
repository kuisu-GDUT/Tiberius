export CUDA_VISIBLE_DEVICES=0
TIBERIUS_ROOT=/data/sukui_data/01_data/02_project_data/Tiberius
#DATA_ROOT=/data/sukui_data/01_data/01_genomics_data/gene_structure

cd $TIBERIUS_ROOT
python bin/train_in_multispecies.py \
  --data /home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic/homo_sapiens_tiberius_20K \
  --out ./output/gorilla_15class_20k \
  --learnMSA . \
  --val_data /home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic/homo_sapiens_tiberius_20K