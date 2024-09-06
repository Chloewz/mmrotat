# SODAA-DOTA S2ANET

python tools/train.py --config configs/s2anet/s2anet_r50_fpn_1x_sodaa_le135.py --work-dir /mnt/d/exp/sodaa_sob/0905 --seed 23

# SODAA S2ANET

python tools/train.py --config configs/sodaa-benchmarks/s2anet_r50_fpn_1x.py --work-dir /mnt/d/exp/sodaa_sob/0906_2 --seed 23

# SODAA S2ANET NO Validation
## Train
python tools/train.py --config configs/sodaa-benchmarks/s2anet_r50_fpn_1x.py --work-dir /mnt/d/exp/sodaa_sob/0906_noval --no-validate --seed 23

## Test
python tools/test.py --config configs/sodaa-benchmarks/s2anet_r50_fpn_1x.py --checkpoint /mnt/d/exp/sodaa_sob/0906_noval/latest.pth --eval mAP