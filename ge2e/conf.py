# nnet opts

lstm_conf = {"num_layers": 3, "hidden_size": 738, "dropout": 0.2}

nnet_conf = {"feature_dim": 64, "embedding_dim": 128, "lstm_conf": lstm_conf}

# trainer opts
opt_kwargs = {"lr": 1e-2, "weight_decay": 1e-5, "momentum": 0.8}

trainer_conf = {
    "optimizer": "sgd",
    "optimizer_kwargs": opt_kwargs,
    "clip_norm": 10,
    "min_lr": 1e-8,
    "patience": 2,
    "factor": 0.5,
    "no_impr": 6,
    "logging_period": 200  # steps
}

# train_dir = "/home/vano/wrkdir/projects_data/sre_2019/toy_dataset/"
# dev_dir = "/home/vano/wrkdir/projects_data/sre_2019/toy_dataset/"
train_dir = "/home/ivank/wrkdir/projects_data/sre_2019/swbd_sre_small_fbank/"
dev_dir = "/home/ivank/wrkdir/projects_data/sre_2019/swbd_sre_small_fbank/"