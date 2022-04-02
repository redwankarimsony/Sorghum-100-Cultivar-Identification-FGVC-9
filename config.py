import os
import torch


DEBUG = False

class CFG:
    data_dir = "data"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    class_mapping_file = os.path.join(data_dir, "train_cultivar_mapping.csv")
    seed = 42
    model_name = 'tf_efficientnet_b3_ns'
    pretrained = True
    img_size = 512
    num_classes = 100
    lr = 1e-4
    max_lr = 1e-3
    pct_start = 0.2
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    num_epochs = 40
    batch_size = 16
    accum = 1
    precision = 16
    n_fold = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')