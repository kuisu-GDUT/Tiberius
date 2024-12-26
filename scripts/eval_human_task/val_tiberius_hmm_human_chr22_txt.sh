export CUDA_VISIBLE_DEVICES=0
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius
#DATA_ROOT=/data/sukui_data/01_data/01_genomics_data/gene_structure
WEIGHT_TIBERIUS=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/weights/tiberius_weights
#WEIGHT_HUMAN_134=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/outputs/Human_15class_tfrecord/epoch_134
WEIGHT_HUMAN_TFRecord_95=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/outputs/train_homo_sapiens_tiberius_pkls_from_tfrecord_10K/epoch_94
cd $TIBERIUS_ROOT

python test_data/val_model_chr_txt.py \
  --model $WEIGHT_TIBERIUS \
  --batch_size 1 \
  --max_length 9999 \
  --learnMSA . \
  --fasta_root /home/share/huadjyin/home/yinpeng/refseq/model_species_tiberius/Homo_sapiens/fasta \
  --label_root /home/share/huadjyin/home/yinpeng/refseq/model_species_tiberius/Homo_sapiens/anno_tiberius \
  --csv_path /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/test_data/Homo_splice_seq_path_v2.csv \
  --save_path ./outputs/eval_two_step_tiberius

#python test_data/val_model_pkl.py \
#  --model ./outputs/Human_15class_tfrecord/epoch_10 \
#  --batch_size 10 \
#  --learnMSA . \
#  --val_data_path /home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic

## eval exo data
#python test_data/val_model_pkl.py \
#  --model $WEIGHT_TIBERIUS \
#  --batch_size 100 \
#  --learnMSA . \
#  --val_data_path /home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/multi_species/intergenic \
#  --val_data_name homo_sapiens_gene_exo_intro \
#  --save_path ./outputs/eval_human_gene_exo