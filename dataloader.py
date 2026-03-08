from torchvision import transforms
from torch.utils.data import DataLoader
from utils.process import (
    RandCrop,
    ToTensor,
    RandHorizontalFlip,
    Normalize,
    five_point_crop,
    random_crop,
)
from utils.process import (
    split_dataset_live,
    split_dataset_tid2013,
    split_dataset_kadid10k,
    split_dataset_csiq,
    split_dataset_koniq10k,
    split_dataset_livec,
    split_dataset_livefb,
    split_dataset_aadb,
)
from utils.charm_preprocess import ImportanceAwareCrop
from data.tid2013 import Tid2013
from data.live import Live
from data.kadid10k import Kadid10k
from data.csiq import CSIQ
from data.koniq10k import Koniq10k
from data.livec import LiveC
from data.livefb import LiveFB
from data.aadb import AADB
from data.mix_dataset import MixDataset
import logging
from utils.color import BOLD, ENDC, GREEN


def _make_train_transform(config):
    """Build train-time transform pipeline, using ImportanceAwareCrop if --charm_crop is set."""
    if getattr(config, "charm_crop", False):
        crop = ImportanceAwareCrop(config.crop_size)
    else:
        crop = RandCrop(config.crop_size)
    return transforms.Compose([crop, Normalize(0.5, 0.5), RandHorizontalFlip(), ToTensor()])


def _make_val_transform():
    """Build validation-time transform pipeline (no cropping)."""
    return transforms.Compose([Normalize(0.5, 0.5), ToTensor()])


_SPLIT_FNS = {
    "tid2013": split_dataset_tid2013,
    "live": split_dataset_live,
    "kadid10k": split_dataset_kadid10k,
    "csiq": split_dataset_csiq,
    "koniq10k": split_dataset_koniq10k,
    "livec": split_dataset_livec,
    "livefb": split_dataset_livefb,
    "aadb": split_dataset_aadb,
}


def _get_paths(dataset_name, root_dir):
    """Return (ref_path, dis_path, txt_file) for a single dataset name."""
    if dataset_name == "tid2013":
        return (
            f"{root_dir}/data/datasets/TID2013/reference_images",
            f"{root_dir}/data/datasets/TID2013/distorted_images",
            f"{root_dir}/data/tid2013_label.txt",
        )
    elif dataset_name == "live":
        return (
            f"{root_dir}/data/datasets/LIVE",
            f"{root_dir}/data/datasets/LIVE",
            f"{root_dir}/data/live_label.txt",
        )
    elif dataset_name == "kadid10k":
        return (
            f"{root_dir}/data/datasets/KADID10K/reference_images",
            f"{root_dir}/data/datasets/KADID10K/distorted_images",
            f"{root_dir}/data/kadid10k_label.txt",
        )
    elif dataset_name == "csiq":
        return (
            f"{root_dir}/data/datasets/CSIQ/src_imgs",
            f"{root_dir}/data/datasets/CSIQ/dst_imgs",
            f"{root_dir}/data/csiq_label.txt",
        )
    elif dataset_name == "koniq10k":
        return (
            None,
            f"{root_dir}/data/datasets/KONIQ/koniq10k_1024x768",
            f"{root_dir}/data/koniq10k_label.txt",
        )
    elif dataset_name == "livec":
        return (
            None,
            f"{root_dir}/data/datasets/LIVEC/Images",
            f"{root_dir}/data/LIVEC_label.txt",
        )
    elif dataset_name == "livefb":
        return (
            None,
            f"{root_dir}/data/datasets/LIVEFB/images",
            f"{root_dir}/data/livefb_labels.txt",
        )
    elif dataset_name == "aadb":
        return (
            None,
            f"{root_dir}/data/datasets/AADB/datasetImages_warp256",
            f"{root_dir}/data/datasets/AADB/imgListFiles_label/imgListTrainRegression_score.txt",
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _prepare_multi_dataset(config, dataset_names, ratio):
    train_transform = _make_train_transform(config)
    val_transform = _make_val_transform()

    train_dataset = MixDataset(transform=train_transform)
    val_dataset = MixDataset(transform=val_transform)

    for name in dataset_names:
        ref_path, dis_path, txt_file = _get_paths(name, config.root_dir)
        train_split, val_split = _SPLIT_FNS[name](
            txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset.add_dataset(ref_path or "", dis_path, txt_file, train_split, name)
        if val_split:
            val_dataset.add_dataset(ref_path or "", dis_path, txt_file, val_split, name)

    return (
        train_dataset if train_dataset.index else [],
        val_dataset if val_dataset.index else [],
    )


def prepare_dataset(config, cross_check=False, ratio=0.8):
    # data load
    if cross_check:
        cross_check_dataset = (config.cross_check_dataset).split(" ")
        if config.dataset in cross_check_dataset:
            ratio = 0
            print(
                BOLD
                + GREEN
                + "--> Dataset [{}] in Cross Check mode! Training ratio is {:.2f}, all will be validation set.".format(
                    config.dataset, ratio
                )
                + ENDC
            )
        else:
            ratio = 1
            print(
                BOLD
                + GREEN
                + "--> Dataset [{}] in Cross Check mode! Training ratio is {:.2f}, all will be training set".format(
                    config.dataset, ratio
                )
                + ENDC
            )
    else:
        print(
            BOLD
            + GREEN
            + "--> Dataset [{}] in Normal mode! Training ratio is {}.".format(
                config.dataset, ratio
            )
            + ENDC
        )

    datasets = config.dataset.split()
    if len(datasets) > 1:
        return _prepare_multi_dataset(config, datasets, ratio)

    if config.dataset == "tid2013":
        train_split, val_split = split_dataset_tid2013(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = Tid2013(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=_make_train_transform(config),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = Tid2013(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=_make_val_transform(),
            )
        else:
            val_dataset = []
    elif config.dataset == "live":
        train_split, val_split = split_dataset_live(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = Live(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=_make_train_transform(config),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = Live(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=_make_val_transform(),
            )
        else:
            val_dataset = []
    elif config.dataset == "kadid10k":
        train_split, val_split = split_dataset_kadid10k(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = Kadid10k(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=_make_train_transform(config),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = Kadid10k(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=_make_val_transform(),
            )
        else:
            val_dataset = []
    elif config.dataset == "csiq":
        train_split, val_split = split_dataset_csiq(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = CSIQ(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=_make_train_transform(config),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = CSIQ(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=_make_val_transform(),
            )
        else:
            val_dataset = []
    elif config.dataset == "koniq10k":
        train_split, val_split = split_dataset_koniq10k(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = Koniq10k(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=_make_train_transform(config),
                resize=True,
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = Koniq10k(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=_make_val_transform(),
                resize=True,
            )
        else:
            val_dataset = []
    elif config.dataset == "livec":
        train_split, val_split = split_dataset_livec(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = LiveC(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=_make_train_transform(config),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = LiveC(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=_make_val_transform(),
            )
        else:
            val_dataset = []
    elif config.dataset == "livefb":
        train_split, val_split = split_dataset_livefb(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = LiveFB(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=_make_train_transform(config),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = LiveFB(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=_make_val_transform(),
            )
        else:
            val_dataset = []
    elif config.dataset == "aadb":
        train_split, val_split = split_dataset_aadb(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = AADB(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=_make_train_transform(config),
            )
        else:
            train_dataset = []
        if val_split:
            # Val split reads from the test file
            val_txt = config.txt_file.replace("Train", "Test")
            val_dataset = AADB(
                dis_path=config.dis_path,
                txt_file_name=val_txt,
                list_name=val_split,
                transform=_make_val_transform(),
            )
        else:
            val_dataset = []
    else:
        raise Exception("dataset not valid")

    logging.info("number of train scenes: {}".format(len(train_dataset)))
    logging.info("number of val scenes: {}".format(len(val_dataset)))

    return train_dataset, val_dataset


def prepare_dataloader(config, cross_check=False):
    train_dataset, val_dataset = prepare_dataset(config, cross_check=cross_check)

    if train_dataset:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=False,
            shuffle=True,
        )
    else:
        train_loader = None
    eval_batch = 1 if "live" in config.dataset else config.batch_size
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=eval_batch,  # config.batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=False,
    )

    print(
        BOLD
        + GREEN
        + "Dataset: {}, train length: {} val length: {}".format(
            config.dataset, len(train_dataset), len(val_dataset)
        )
        + ENDC
    )

    return train_loader, val_loader
