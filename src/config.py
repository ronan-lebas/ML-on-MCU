config = {
    'use_wandb': True,
    'num_epochs': 100,
    'batch_size': 128,
    'max_sample_per_class': -1,
    'only_classes': [
        "yes",
        "no",
        "on",
        "off",
        "up",
        "down",
        "left",
        "right"
    ],
    'sampling_rate': 8000,
    'optimizer': 'SGD',
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'lr_scheduler': 'CosineAnnealing',
    'n_mfcc': 40,
    'melkwargs': {
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 64
    },
}