# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
from statistics import mode

import torch
from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info
from mmengine.analysis import (specific_stats_str,
                               specific_stats_table,
                               FlopCountAnalysis,
                               parameter_count,
                               ActivationCountAnalysis)
from mmengine.analysis.statistics_helper import _format_size
from mmengine import print_log

from mmcls.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_classifier(cfg.model)
    model.eval()

    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    inputs = (torch.randn((1, *input_shape)), )

    flops_ = FlopCountAnalysis(model, inputs)
    activations_ = ActivationCountAnalysis(model, inputs)

    flops = _format_size(flops_.total())
    activations = _format_size(activations_.total())
    params = _format_size(parameter_count(model)[''])

    model_str = specific_stats_str(flops=flops_, activations=activations_)
    model_table = specific_stats_table(
        flops=flops_,
        activations=activations_,
        show_param_shapes=True,
    )

    print_log('\n'+model_str)
    print_log('\n'+model_table)

    split_line = '=' * 30
    print_log(f'\n{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\nActivation: {activations}\n{split_line}')

    print_log('Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.', 'current', logging.WARNING)


if __name__ == '__main__':
    main()
