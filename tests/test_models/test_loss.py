# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import pytest
import torch
import torch.nn.functional as F

from mmrotate import digit_version
from mmrotate.models.losses import (BCConvexGIoULoss, ConvexGIoULoss, GDLoss,
                                    GDLoss_v1, KFLoss, KLDRepPointsLoss,
                                    RotatedIoULoss)
from mmrotate.models.losses import LSFocalLoss


# @pytest.mark.skipif(
#     not torch.cuda.is_available(), reason='requires CUDA support')
# @pytest.mark.parametrize('loss_class',
#                          [BCConvexGIoULoss, ConvexGIoULoss, KLDRepPointsLoss])
# def test_convex_regression_losses(loss_class):
#     """Tests convex regression losses.

#     Args:
#         loss_class (str): type of convex loss.
#     """
#     pred = torch.rand((10, 18)).cuda()
#     target = torch.rand((10, 8)).cuda()
#     weight = torch.rand((10, )).cuda()

#     # Test loss forward
#     loss = loss_class()(pred, target)
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with weight
#     loss = loss_class()(pred, target, weight)
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with reduction_override
#     loss = loss_class()(pred, target, reduction_override='mean')
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with avg_factor
#     loss = loss_class()(pred, target, avg_factor=10)
#     assert isinstance(loss, torch.Tensor)


# # @pytest.mark.skipif(
# #     not torch.cuda.is_available(), reason='requires CUDA support')
# @pytest.mark.parametrize('loss_type',
#                          ['gwd', 'kld', 'jd', 'kld_symmax', 'kld_symmin'])
# def test_gaussian_regression_losses(loss_type):
#     """Tests gaussian regression losses.

#     Args:
#         loss_class (str): type of gaussian loss.
#     """
#     pred = torch.rand((10, 5))
#     target = torch.rand((10, 5))
#     weight = torch.rand((10, 5))

#     # Test loss forward with weight
#     loss = GDLoss(loss_type)(pred, target, weight)
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with reduction_override
#     loss = GDLoss(loss_type)(pred, target, weight, reduction_override='mean')
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with avg_factor
#     loss = GDLoss(loss_type)(pred, target, weight, avg_factor=10)
#     assert isinstance(loss, torch.Tensor)


# # @pytest.mark.skipif(
# #     not torch.cuda.is_available(), reason='requires CUDA support')
# # @pytest.mark.parametrize('loss_type', ['bcd', 'kld', 'gwd'])
# def test_gaussian_v1_regression_losses(loss_type):
#     """Tests gaussian regression losses v1.

#     Args:
#         loss_class (str): type of gaussian loss v1.
#     """
#     pred = torch.rand((10, 5))
#     target = torch.rand((10, 5))
#     weight = torch.rand((10, 5))

#     # Test loss forward with weight
#     loss = GDLoss_v1(loss_type)(pred, target, weight)
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with reduction_override
#     loss = GDLoss_v1(loss_type)(
#         pred, target, weight, reduction_override='mean')
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with avg_factor
#     loss = GDLoss_v1(loss_type)(pred, target, weight, avg_factor=10)
#     assert isinstance(loss, torch.Tensor)


# # @pytest.mark.skipif(
# #     not torch.cuda.is_available(), reason='requires CUDA support')
# def test_kfiou_regression_losses():
#     """Tests kfiou regression loss."""
#     pred = torch.rand((10, 5))
#     target = torch.rand((10, 5))
#     weight = torch.rand((10, 5))
#     pred_decode = torch.rand((10, 5))
#     targets_decode = torch.rand((10, 5))

#     # Test loss forward with weight
#     loss = KFLoss()(
#         pred,
#         target,
#         weight,
#         pred_decode=pred_decode,
#         targets_decode=targets_decode)
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with reduction_override
#     loss = KFLoss()(
#         pred,
#         target,
#         weight,
#         pred_decode=pred_decode,
#         targets_decode=targets_decode,
#         reduction_override='mean')
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with avg_factor
#     loss = KFLoss()(
#         pred,
#         target,
#         weight,
#         pred_decode=pred_decode,
#         targets_decode=targets_decode,
#         avg_factor=10)
#     assert isinstance(loss, torch.Tensor)


# @pytest.mark.skipif(
#     not torch.cuda.is_available(), reason='requires CUDA support')
# @pytest.mark.skipif(
#     digit_version(mmcv.__version__) <= digit_version('1.5.0'),
#     reason='requires mmcv>=1.5.0')
# def test_rotated_iou_losses():
#     """Tests convex regression losses."""
#     pred = torch.rand((10, 5)).cuda()
#     target = torch.rand((10, 5)).cuda()
#     weight = torch.rand((10, )).cuda()

#     # Test loss mode
#     loss = RotatedIoULoss(linear=True)(pred, target)
#     assert isinstance(loss, torch.Tensor)

#     loss = RotatedIoULoss(mode='linear')(pred, target)
#     assert isinstance(loss, torch.Tensor)

#     loss = RotatedIoULoss(mode='log')(pred, target)
#     assert isinstance(loss, torch.Tensor)

#     loss = RotatedIoULoss(mode='square')(pred, target)
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward
#     loss = RotatedIoULoss()(pred, target)
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with weight
#     loss = RotatedIoULoss()(pred, target, weight)
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with reduction_override
#     loss = RotatedIoULoss()(pred, target, reduction_override='mean')
#     assert isinstance(loss, torch.Tensor)

#     # Test loss forward with avg_factor
#     loss = RotatedIoULoss()(pred, target, avg_factor=10)
#     assert isinstance(loss, torch.Tensor)

# @pytest.mark.skipif(
#     not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('loss_class', [LSFocalLoss])
@pytest.mark.parametrize('input_shape',[(10,5), (3,5,40,40)])
def test_label_smoothing_focal_losses(loss_class, input_shape):
    pred = torch.rand(input_shape)
    target = torch.randint(0, 5, (input_shape[0], ))
    if len(input_shape) == 4:
        B, N, W, H = input_shape
        target = F.one_hot(torch.randint(0, 5, (B * W * H, )),
                           5).reshape(B, W, H, N).permute(0, 3, 1, 2)

    pred_sigmoid = pred.sigmoid()

    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)


    loss = loss_class()(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with reduction_override
    loss = loss_class()(pred, target, reduction_override='mean')
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = loss_class()(pred, target, avg_factor=10)
    assert isinstance(loss, torch.Tensor)

    with pytest.raises(ValueError):
        # loss can evaluate with avg_factor only if
        # reduction is None, 'none' or 'mean'.
        reduction_override = 'sum'
        loss_class()(
            pred, target, avg_factor=10, reduction_override=reduction_override)

    # Test loss forward with avg_factor and reduction
    for reduction_override in [None, 'none', 'mean']:
        loss_class()(
            pred, target, avg_factor=10, reduction_override=reduction_override)
        assert isinstance(loss, torch.Tensor)

