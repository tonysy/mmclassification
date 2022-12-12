_base_ = [
    '../_base_/models/vit-base-p32.py',
    '../_base_/datasets/imagenet_bs64_pil_resize_autoaug.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# model setting
model = dict(
    backbone=dict(pre_norm=True,),
    # head=dict(hidden_dim=3072),
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
)

# schedule setting
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[
        0.48145466*255,
        0.4578275*255,
        0.40821073*255
    ],
    std=[
        0.26862954*255,
        0.26130258*255,
        0.27577711*255
    ],
    # convert image from BGR to RGB
    to_rgb=True,
)
