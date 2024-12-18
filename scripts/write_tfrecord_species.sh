export CUDA_VISIBLE_DEVICES=2
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius

cd $TIBERIUS_ROOT
python bin/write_tfrecord_species.py \
  --gtf /home/share/huadjyin/home/yinpeng/sukui_data/gene_structure/Species/Mus_musculus/Mus_musculus.longest.tiberius.gtf \
  --fasta /home/share/huadjyin/home/yinpeng/sukui_data/gene_structure/Species/Mus_musculus/GCF_000001635.26_GRCm38.p6_genomic.fna \
  --wsize 9999 \
  --out /home/share/huadjyin/home/yinpeng/sukui_data/gene_structure/Tiberius/pkls/Mus_musculus_custom_hmm_10K/Mus_musculus_hmm \
  --transition \
  --pkl