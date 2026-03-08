import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # root directory
    parser.add_argument("--root_dir", type=str, default="/your/root/dir")
    # model
    parser.add_argument(
        "--network", type=str, default="model", help="model file name of your network"
    )
    parser.add_argument("--GPU", type=int, default=0)
    # optimization
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--T_max", type=int, default=50)
    parser.add_argument("--eta_min", type=int, default=0)
    parser.add_argument("--num_avg_val", type=int, default=20)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    # saving
    parser.add_argument(
        "--model_name", type=str, default="model", help="your checkpoint saving name"
    )
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--log_file", type=str, default="log.txt")
    parser.add_argument("--save_all", action="store_true")  # save each stage result
    # dataset related
    parser.add_argument("--dataset", type=str, default="tid2013")
    # others
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--random_seed", action="store_true")
    parser.add_argument(
        "--save_sh", type=str, default="./train.sh", help="your current sh file"
    )
    parser.add_argument("--testing", action="store_true")
    parser.add_argument(
        "--checkpoint", type=str, default="./ckpt", help="your iqa model checkpoint"
    )
    parser.add_argument("--cross_check", action="store_true")
    parser.add_argument("--cross_check_dataset", type=str, default="live tid2013 csiq")
    parser.add_argument("--training_mode", type=str, default="", help="mix/FR/NR")
    parser.add_argument("--infer_mode", type=str, default="", help="FR/NR")
    parser.add_argument("--log_interval", type=int, default=20, help="log every N training steps")
    # Charm + DINOv3 backbone options
    parser.add_argument(
        "--backbone",
        type=str,
        default="facebook/dinov3-convnext-base-pretrain-lvd1689m",
        help="HuggingFace model name for ConvNeXt backbone (used with --network model_charm)",
    )
    parser.add_argument(
        "--charm_crop",
        action="store_true",
        help="Use importance-aware cropping instead of random cropping",
    )
    parser.add_argument(
        "--freeze_backbone_stages",
        type=int,
        default=0,
        help="Freeze first N ConvNeXt stages (0-4). 2 is recommended for DINOv3.",
    )
    parser.add_argument(
        "--lr_decay_factor",
        type=float,
        default=1.0,
        help="Layer-wise LR decay factor for backbone stages (e.g. 0.65). "
             "Stage i gets lr * factor^(num_stages - 1 - i). 1.0 = no decay.",
    )
    parser.add_argument(
        "--keep_checkpoints",
        type=int,
        default=0,
        help="Keep only the latest N checkpoints per mode (FR/NR). 0 = keep all.",
    )

    return parser.parse_args()


def make_template(config):
    # common operations
    datasets = config.dataset.split()
    config.output_path = "{}/{}".format(config.output_path, "_".join(datasets))

    # datset related
    # synthetic
    config.root_dir = (
        config.root_dir[:-1]
        if config.root_dir[-2] != "./" and config.root_dir[-1] == "/"
        else config.root_dir
    )

    # For single-dataset runs, set the flat path attributes used by the dataloader.
    # For multi-dataset runs these are computed per-dataset inside prepare_dataset.
    if config.dataset == "tid2013":
        config.ref_path = f"{config.root_dir}/data/datasets/TID2013/reference_images"
        config.dis_path = f"{config.root_dir}/data/datasets/TID2013/distorted_images"
        config.txt_file = f"{config.root_dir}/data/tid2013_label.txt"
    elif config.dataset == "live":
        config.ref_path = f"{config.root_dir}/data/datasets/LIVE"
        config.dis_path = f"{config.root_dir}/data/datasets/LIVE"
        config.txt_file = f"{config.root_dir}/data/live_label.txt"
    elif config.dataset == "kadid10k":
        config.ref_path = f"{config.root_dir}/data/datasets/KADID10K/reference_images"
        config.dis_path = f"{config.root_dir}/data/datasets/KADID10K/distorted_images"
        config.txt_file = f"{config.root_dir}/data/kadid10k_label.txt"
    elif config.dataset == "csiq":
        config.ref_path = f"{config.root_dir}/data/datasets/CSIQ/src_imgs"
        config.dis_path = f"{config.root_dir}/data/datasets/CSIQ/dst_imgs"
        config.txt_file = f"{config.root_dir}/data/csiq_label.txt"
    # authentic
    elif config.dataset == "koniq10k":
        config.dis_path = f"{config.root_dir}/data/datasets/KONIQ/koniq10k_1024x768"
        config.txt_file = f"{config.root_dir}/data/koniq10k_label.txt"
    elif config.dataset == "livec":
        config.dis_path = f"{config.root_dir}/data/datasets/LIVEC/Images"
        config.txt_file = f"{config.root_dir}/data/LIVEC_label.txt"
    elif config.dataset == "livefb":
        config.dis_path = f"{config.root_dir}/data/datasets/LIVEFB/images"
        config.txt_file = f"{config.root_dir}/data/livefb_labels.txt"
    elif config.dataset == "aadb":
        config.dis_path = f"{config.root_dir}/data/datasets/AADB/datasetImages_warp256"
        config.txt_file = f"{config.root_dir}/data/datasets/AADB/imgListFiles_label/imgListTrainRegression_score.txt"


def get_option():
    config = parse_args()
    make_template(config)
    return config


def cross_check_template(dataset, config):
    # datset related
    # synthetic
    if dataset == "tid2013":
        config.ref_path = f"{config.root_dir}/data/datasets/TID2013/reference_images"
        config.dis_path = f"{config.root_dir}/data/datasets/TID2013/distorted_images"
        config.txt_file = f"{config.root_dir}/data/tid2013_label.txt"
    elif dataset == "live":
        config.ref_path = f"{config.root_dir}/data/datasets/LIVE"
        config.dis_path = f"{config.root_dir}/data/datasets/LIVE"
        config.txt_file = f"{config.root_dir}/data/live_label.txt"
    elif dataset == "kadid10k":
        config.ref_path = f"{config.root_dir}/data/datasets/KADID10K/reference_images"
        config.dis_path = f"{config.root_dir}/data/datasets/KADID10K/distorted_images"
        config.txt_file = f"{config.root_dir}/data/kadid10k_label.txt"
    elif dataset == "csiq":
        config.ref_path = f"{config.root_dir}/data/datasets/CSIQ/src_imgs"
        config.dis_path = f"{config.root_dir}/data/datasets/CSIQ/dst_imgs"
        config.txt_file = f"{config.root_dir}/data/csiq_label.txt"
    # authentic
    elif dataset == "koniq10k":
        config.dis_path = f"{config.root_dir}/data/datasets/KONIQ/koniq10k_1024x768"
        config.txt_file = f"{config.root_dir}/data/koniq10k_label.txt"
    elif dataset == "livec":
        config.dis_path = f"{config.root_dir}/data/datasets/LIVEC/Images"
        config.txt_file = f"{config.root_dir}/data/LIVEC_label.txt"
    elif dataset == "livefb":
        config.dis_path = f"{config.root_dir}/data/datasets/LIVEFB/images"
        config.txt_file = f"{config.root_dir}/data/livefb_labels.txt"
    elif dataset == "aadb":
        config.dis_path = f"{config.root_dir}/data/datasets/AADB/datasetImages_warp256"
        config.txt_file = f"{config.root_dir}/data/datasets/AADB/imgListFiles_label/imgListTrainRegression_score.txt"
