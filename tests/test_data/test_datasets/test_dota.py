# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import tempfile

import numpy as np
import pytest
from mmdet.datasets import build_dataset

from mmrotate.datasets.sodaa import SODAADataset


def _create_dummy_results():
    """Create dummy results."""
    boxes = [
        np.array([[4.3150e+02, 7.0600e+02, 6.7686e+01, 2.1990e+01, 2.9842e-02],
                  [5.6351e+02, 5.3575e+02, 1.0018e+02, 1.8971e+01, 5.5499e-02],
                  [5.7450e+02, 5.8450e+02, 9.5567e+01, 2.1094e+01,
                   8.4012e-02]])
    ]
    return [boxes]


@pytest.mark.parametrize('angle_version', ['oc'])
def test_dota_dataset(angle_version):
    """Test DOTA dataset.

    Args:
        angle_version (str, optional): Angle representations.
    """
    # test CLASSES
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    data_config = dict(
        type=SODAADataset,
        version=angle_version,
        ann_file='/mnt/d/exp/sodaa_sob/datasets-small/labels/',
        img_prefix='/mnt/d/exp/sodaa_sob/datasets-small/images/',
        pipeline=train_pipeline)
    dataset = build_dataset(data_config)
    # assert dataset.CLASSES == ('plane', 'baseball-diamond', 'bridge',
    #                            'ground-track-field', 'small-vehicle',
    #                            'large-vehicle', 'ship', 'tennis-court',
    #                            'basketball-court', 'storage-tank',
    #                            'soccer-ball-field', 'roundabout', 'harbor',
    #                            'swimming-pool', 'helicopter')
    assert dataset.CLASSES == ('airplane', 'helicopter', 'small-vehicle', 'large-vehicle', 
                               'ship', 'container', 'storage-tank', 'swimming-pool','windmill')

    # test eval
    dataset.CLASSES = ('airplane', )
    fake_results = _create_dummy_results()
    eval_results = dataset.evaluate(fake_results)
    np.testing.assert_almost_equal(eval_results['mAP'], 0.7272727)

    # test format_results
    tmp_filename = osp.join(tempfile.gettempdir(), 'merge_results')
    if osp.exists(tmp_filename):
        shutil.rmtree(tmp_filename)
    dataset.format_results(fake_results, submission_dir=tmp_filename)
    shutil.rmtree(tmp_filename)

    # test filter_empty_gt=False
    full_data_config = dict(
        type=SODAADataset,
        version=angle_version,
        ann_file='/mnt/d/exp/sodaa_sob/datasets-small/labels/',
        img_prefix='/mnt/d/exp/sodaa_sob/datasets-small/images/',
        pipeline=train_pipeline,
        filter_empty_gt=False)
    full_dataset = build_dataset(full_data_config)
    assert len(dataset) == 1 and len(full_dataset) == 2


# if __name__ == '__main__':
#     test_dota_dataset(['oc'])