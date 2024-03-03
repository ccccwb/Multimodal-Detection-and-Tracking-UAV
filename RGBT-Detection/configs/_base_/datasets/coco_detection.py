dataset_type = 'CocoDataset'
data_root = '/media/data3/caiwb/RGBTDet/data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOBBAnnotations', with_bbox=True,
         with_label=True),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),###
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomOBBRotate', rotate_after_flip=True,
         angles=(0, 0), vert_rate=0.5, vert_cls=['roundabout', 'storage-tank']),
    dict(type='Pad', size_divisor=32),
    dict(type='DOTASpecialIgnore', ignore_size=2),
    dict(type='FliterEmpty'),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(type='OBBCollect', keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[(640, 512)],
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='OBBRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomOBBRotate', rotate_after_flip=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img']),
        ])
]
# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=8,
#     train=dict(
#         type=dataset_type,
#         #rgb
#         ann_file=data_root + 'annotations/train.json',
#         img_prefix=data_root + 'train2017/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         #rgb
#         ann_file=data_root + 'annotations/val.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         #rgb
#         ann_file=data_root + 'annotations/val.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline
#         ))
# evaluation = None
evaluation = dict(metric='mAP')
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        #rgb
        ann_file=data_root + 'annotations/train_mod.json',#train_mod
        img_prefix=data_root + 'DV2/train/trainimg_mod/',#trainimg_mod
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        #rgb
        ann_file=data_root + 'annotations/val_mod.json',#val_mod
        img_prefix=data_root + 'DV2/val/valimg_mod/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        #rgb
        ann_file=data_root + 'annotations/val_mod.json',#val_mod
        img_prefix=data_root + 'DV2/val/valimg_mod/',
        pipeline=test_pipeline)
        )