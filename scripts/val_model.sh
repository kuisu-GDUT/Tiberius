export CUDA_VISIBLE_DEVICES=2
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius
#DATA_ROOT=/data/sukui_data/01_data/01_genomics_data/gene_structure

cd $TIBERIUS_ROOT

python test_data/val_model.py \
  --model ./outputs/Human_15class_tfrecord/epoch_134 \
  --batch_size 64 \
  --learnMSA . \
  --val_data_path ./outputs/Human_15class_tfrecord/validation_data_10K.npz

#python test_data/val_model_pkl.py \
#  --model ./outputs/Human_15class_tfrecord/epoch_134 \
#  --batch_size 16 \
#  --learnMSA . \
#  --val_data_path /home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic