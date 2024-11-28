export CUDA_VISIBLE_DEVICES=2
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius

cd $TIBERIUS_ROOT

python test_data/test_vit.py \
  --annot ./test_data/annot_longest_fixed_paper/Homo_sapiens.gtf \
  --genome ./test_data/annot_longest_fixed_paper/Homo_sapiens_GCF_000001405.40_GRCh38.p14_genomic.fna \
  --model_lstm weights/tiberius_weights/ \
  --learnMSA . \
  --batch_size 2 \
  --seq_len 9999 \
  --out_dir ./outputs/test_vit