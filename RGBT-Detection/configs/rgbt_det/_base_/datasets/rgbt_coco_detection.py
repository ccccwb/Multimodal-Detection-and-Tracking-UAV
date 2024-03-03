dataset_type = 'RgbtCocoDataset'
data_root = '/media/data3/caiwb/RGBTDetection/data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadRGBTImageFromFile'),
    dict(type='LoadRGBTAnnotations', with_bbox=True,
         with_label=True),
    dict(type='RGBTResize', img_scale=(640, 512), keep_ratio=True),###
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(type='RGBTNormalize', **img_norm_cfg),
    dict(type='RandomRGBTRotate', rotate_after_flip=True,
         angles=(0, 0), vert_rate=0.5, vert_cls=['roundabout', 'storage-tank']),
    dict(type='RGBTPad', size_divisor=32),
    # dict(type='DOTASpecialIgnore', ignore_size=2),
    dict(type='RGBTFliterEmpty'),
    dict(type='RGBTMask2OBB', obb_type='obb'),
    dict(type='RGBTDefaultFormatBundle'),
    dict(type='RGBTCollect', keys=['img_r', 'gt_bboxes_r', 'gt_obboxes_r', 'gt_labels_r', 'img_i', 'gt_bboxes_i', 'gt_obboxes_i', 'gt_labels_i']) #, 'img_i', 'gt_bboxes_i', 'gt_obboxes_i', 'gt_labels_i'
]
test_pipeline = [
    dict(type='LoadRGBTImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[(640, 512)],
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='RGBTResize', keep_ratio=True),
            dict(type='OBBRandomFlip'),
            dict(type='RGBTNormalize', **img_norm_cfg),
            dict(type='RandomRGBTRotate', rotate_after_flip=True),
            dict(type='RGBTPad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img_r', 'img_i']), #, 'img_i'
            dict(type='RGBTCollect', keys=['img_r', 'img_i']), # , 'img_i'
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
    workers_per_gpu=2, 
    train=dict(
        type=dataset_type,
        #rgb
        ann_file_r=data_root + 'annotations/train_mod_illumination.json',#train_mod_illumination
        img_prefix_r=data_root + 'DV2/train/trainimg_mod/',#trainimg_mod
        #红外
        ann_file_i=data_root + 'annotations/trainr_mod_illumination.json',#trainr_mod
        img_prefix_i=data_root + 'DV2/train/trainimgr_mod/',#trainimgr_mod

        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        #rgb
        ann_file_r=data_root + 'annotations/val_mod_illumination.json',#val_mod_illumination
        img_prefix_r=data_root + 'DV2/val/valimg_mod/',
        #红外
        ann_file_i=data_root + 'annotations/valr_mod_illumination.json',
        img_prefix_i=data_root + 'DV2/val/valimgr_mod/',

        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        #rgb
        ann_file_r=data_root + 'annotations/val_mod_illumination.json',#val_mod
        img_prefix_r=data_root + 'DV2/val/valimg_mod/',
        #红外
        ann_file_i=data_root + 'annotations/valr_mod_illumination.json',
        img_prefix_i=data_root + 'DV2/val/valimgr_mod/',

        pipeline=test_pipeline)
        )