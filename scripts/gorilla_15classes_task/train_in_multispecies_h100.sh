export CUDA_VISIBLE_DEVICES=0
TIBERIUS_ROOT=/data/sukui_data/01_data/02_project_data/Tiberius

cd $TIBERIUS_ROOT
python bin/train_in_multispecies.py \
  --data /data/sukui_data/01_data/01_genomics_data/gene_structure \
  --out ./output/gorilla_15class_20k/ \
  --learnMSA . \
  --val_data /data/sukui_data/01_data/01_genomics_data/gene_structure