# SODAA-DOTA S2ANET

python tools/train.py --config configs/s2anet/s2anet_r50_fpn_1x_sodaa_le135.py --work-dir /mnt/d/exp/sodaa_sob/0905 --seed 23

# SODAA S2ANET

python tools/train.py --config configs/sodaa-benchmarks/s2anet_r50_fpn_1x.py --work-dir /mnt/d/exp/sodaa_sob/0906_2 --seed 23

# SODAA-DOTA S2ANET
## Train
python tools/train.py --config configs/sodaa-benchmarks/s2anet_r50_fpn_1x_dota.py --work-dir /mnt/d/exp/sodaa_sob/4060/0909 --seed 23

## Test
python tools/test.py --config configs/sodaa-benchmarks/s2anet_r50_fpn_1x.py --checkpoint /mnt/d/exp/sodaa_sob/a6000result/0907/epoch_9.pth --eval mAP --show-dir /mnt/d/exp/sodaa_sob/a6000result/0907/visualize

# SODAA-DOTA S2ANET SmoothFocalLoss
## Train
python tools/train.py --config configs/sodaa-benchmarks/s2anet_r50_fpn_1x_dota_smooth.py --work-dir /mnt/d/exp/sodaa_sob/4060/0909_smooth --seed 23