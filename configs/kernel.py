import logging

# model settings
model = dict(
    type="kernel_Estimation",
    conv_net = dict(
        type = 'pixelwisekernel',
        kernel_size = 11,
    ),
)


# dataset settings
dataset_type = "RawsrDataset"
data_root =  r"/home/chenyuxiang/repos/00179"

train_pipeline = [
    
]

data = dict(
    samples_per_gpu=1, # 4 * 1024
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        pipeline=train_pipeline,
    ),
)

lr = 0.001
optimizer = dict(type='AdamW', lr=lr, betas=(0.9, 0.99), eps = 1e-15, weight_decay=1e-12)

lr_config = dict(
    type="one_cycle", max_lr=lr, div_factor=10.0, pct_start=0.1, final_div_factor = 1e2
)

checkpoint_config = dict(interval=100)

log_config = dict(
    interval=50000000,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

total_epochs = 10000
log_level = "INFO"
work_dir = "./workdirs/kernel"
load_from = None
resume_from = None
workflow = [("train", 1),]
