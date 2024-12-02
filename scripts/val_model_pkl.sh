export CUDA_VISIBLE_DEVICES=2
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius
#DATA_ROOT=/data/sukui_data/01_data/01_genomics_data/gene_structure
WEIGHT_TIBERIUS=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/weights/tiberius_weights
WEIGHT_HUMAN_134=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/outputs/Human_15class_tfrecord/epoch_134
cd $TIBERIUS_ROOT

#python test_data/val_model_pkl.py \
#  --model $WEIGHT_HUMAN_134 \
#  --batch_size 100 \
#  --learnMSA . \
#  --val_data_path /home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/Tiberius/tfrecords/Homo_sapiens

python test_data/val_model_pkl.py \
  --model ./outputs/Human_15class_tfrecord/epoch_10 \
  --batch_size 10 \
  --learnMSA . \
  --val_data_path /home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic