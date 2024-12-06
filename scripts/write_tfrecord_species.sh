export CUDA_VISIBLE_DEVICES=2
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius

cd $TIBERIUS_ROOT
python bin/write_tfrecord_species.py \
  --gtf ./test_data/annot_longest_fixed_paper/Homo_sapiens.gtf \
  --fasta ./test_data/annot_longest_fixed_paper/Homo_sapiens_GCF_000001405.40_GRCh38.p14_genomic.fna \
  --wsize 19998 \
  --out /home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/Tiberius/pkls/Homo_sapiens_hmm_100K/Homo_sapiens_hmm \
  --transition \
  --pkl