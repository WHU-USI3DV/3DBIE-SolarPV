export NGPUS=2
# train on source data
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/configs/rs_deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_try/

# # train with fine-grained adversarial alignment （FGDAL）
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train_adv.py -cfg configs/rs_deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test resume results/src_r101_try/model_iter020000.pth
# # train with FGDAL-MSF
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train_SEmsf_adv_BD+FD_FreezeBackbone.py -cfg configs/rs_deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test resume results/src_r101_try/model_iter020000.pth
# train with FGDAL-MSF-DNT
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_SEmsf_dnt_adv_BD+FD_FreezeBackbone.py -cfg configs/rs_deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test resume results/src_r101_try/model_iter030000.pth

echo "Optionally, required to modify the network in the code to conduct self distill"
# generate pseudo labels for self distillation
python test.py -cfg configs/rs_deeplabv2_r101_tgt_self_distill.yaml --saveres resume results/adv_test/model_iter080000.pth OUTPUT_DIR datasets/rs/soft_labels DATASETS.TEST rs_train
# train with self distillation
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_self_distill.py -cfg configs/rs_r101_tgt_self_distill.yaml OUTPUT_DIR results/sd_test
