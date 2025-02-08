export CUDA_VISIBLE_DEVICES=3
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius

cd $TIBERIUS_ROOT
python bin/train_in_human.py \
  --genome /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/test_data/hg38_nop56.fasta \
  --out /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/output.gtf \
  --model /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/DNA_LLM/outputs/supervised/FishsTiberius7label_Combine_dataset_with_10_fish_tiberius_transformer \
  --learnMSA /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/ \
  --batch_size 2 \
  --seq_len 9999 \
  --parallel_factor 1