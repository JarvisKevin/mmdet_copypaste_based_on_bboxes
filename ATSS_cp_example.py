_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ATSS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)


# dataset settings
dataset_type = 'CocoDataset'
data_root = './data/coco_val2017/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

image_size = (1333,800)
load_pipeline = [ 
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True), 
    # Not keep_ratio to make sure the consistent image shape in MultiImageMixDataset.
    dict( 
        type='Resize', 
        img_scale=image_size, 
        keep_ratio=False),    
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)), 
    dict(type='RandomFlip', flip_ratio=0.5), 
] 
train_pipeline = [ 
    dict(type='CopyPaste', max_num_pasted=5), 
    dict(type='Normalize', **img_norm_cfg), 
    dict(type='DefaultFormatBundle'), 
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
] 

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train=dict( 
    type='MultiImageMixDataset', 
    dataset=dict( 
        type=dataset_type, 
        ann_file=data_root + 'annotations/instances_val2017.json', 
        img_prefix=data_root + 'images/', 
        pipeline=load_pipeline), 
   pipeline=train_pipeline) 


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=train,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')

