import os
import torch
import numpy as np
import logging
import time
import random
import importlib
from dataloader import prepare_dataloader
from tqdm import tqdm
from options import get_option
from scipy.stats import spearmanr, pearsonr
from utils.process import five_point_crop, random_crop

import wandb


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False


def set_logging(config):
    filename = os.path.join(config.output_path, config.model_name, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode="w",
        format="[%(asctime)s %(levelname)-8s] %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )


def build_optimizer(net, config):
    """Build optimizer with per-stage LR decay and optional stage freezing.

    For the original Swin-T model (--network model) this falls back to the
    simple 2-group split (encoder at 0.1x LR, body at 1x LR).

    For model_charm (ConvNeXt backbone), it creates per-stage param groups
    with layer-wise LR decay and freezes early stages if requested.
    """
    lr = config.learning_rate
    wd = config.weight_decay
    freeze_stages = getattr(config, "freeze_backbone_stages", 0)
    decay = getattr(config, "lr_decay_factor", 1.0)

    # Detect whether we have a staged ConvNeXt backbone
    has_stages = any(
        "encoder.stages." in name for name, _ in net.named_parameters()
    )

    if not has_stages or decay == 1.0 and freeze_stages == 0:
        # Original 2-group behavior
        enc, body = [], []
        for name, param in net.named_parameters():
            if "encoder" in name:
                enc.append(param)
            else:
                body.append(param)
        assert enc, "encoder is empty"
        return torch.optim.Adam(
            [{"params": enc, "lr": lr * 0.1}, {"params": body}],
            lr=lr, weight_decay=wd,
        )

    # Per-stage param groups for ConvNeXt
    # Discover how many stages exist
    stage_ids = set()
    for name, _ in net.named_parameters():
        if "encoder.stages." in name:
            # e.g. "encoder.stages.2.layers.0..."
            stage_ids.add(int(name.split("encoder.stages.")[1].split(".")[0]))
    num_stages = max(stage_ids) + 1

    # Bucket params: stage_params[i] for stage i, body for the rest.
    # Non-stage encoder params (layer_norm, pool) go into the last stage's group.
    stage_params = {i: [] for i in range(num_stages)}
    body = []

    for name, param in net.named_parameters():
        if "encoder.stages." in name:
            stage_i = int(name.split("encoder.stages.")[1].split(".")[0])
            if stage_i < freeze_stages:
                param.requires_grad = False
            else:
                stage_params[stage_i].append(param)
        elif "encoder" in name:
            # Non-stage encoder params (layer_norm, pool) — group with last stage
            stage_params[num_stages - 1].append(param)
        else:
            body.append(param)

    # Build param groups with decayed LR: later stages get higher LR
    # stage i gets: lr * 0.1 * decay^(num_stages - 1 - i)
    param_groups = []
    for i in range(num_stages):
        if stage_params[i]:
            stage_lr = lr * 0.1 * (decay ** (num_stages - 1 - i))
            param_groups.append({"params": stage_params[i], "lr": stage_lr})

    # Body (fusion head) gets full LR
    param_groups.append({"params": body})

    frozen_n = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    trainable_n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Optimizer: {len(param_groups)} param groups, "
          f"{trainable_n/1e6:.1f}M trainable / {frozen_n/1e6:.1f}M frozen, "
          f"lr_decay={decay}, frozen_stages={freeze_stages}")
    for i, pg in enumerate(param_groups):
        n = sum(p.numel() for p in pg["params"])
        print(f"  group {i}: {n/1e6:.1f}M params, lr={pg.get('lr', lr):.2e}")

    return torch.optim.Adam(param_groups, lr=lr, weight_decay=wd)


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader, device, mode="mix", log_interval=50):
    losses = []
    net.train()
    pred_epoch = []
    labels_epoch = []

    cur_lr = scheduler.get_last_lr()

    for step, data in enumerate(tqdm(train_loader)):

        # Legacy 2-group case: override encoder LR = 0.1 * body LR.
        # With per-stage groups the ratios are already baked in.
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[0]["lr"] = cur_lr[1] * 0.1

        x_d = data["d_img_org"].to(device)
        x_r = data["r_img_org"].to(device) if "r_img_org" in data else x_d
        labels = torch.squeeze(data["score"].float()).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            if mode == "mix":
                if random.random() < 0.5:
                    pred_d = net(x_d, x_r, mode="FR")
                else:
                    pred_d = net(x_d, x_d, mode="NR")
            elif mode.upper() == "FR":
                pred_d = net(x_d, x_r)
            elif mode.upper() == "NR":
                pred_d = net(x_d, x_d)

            loss = criterion(torch.squeeze(pred_d), labels)

        optimizer.zero_grad()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        pred_epoch = np.append(pred_epoch, pred_d.data.cpu().numpy())
        labels_epoch = np.append(labels_epoch, labels.data.cpu().numpy())

        if log_interval > 0 and (step + 1) % log_interval == 0:
            avg_loss = np.mean(losses[-log_interval:])
            wandb.log({"train/step_loss": avg_loss, "train/lr": cur_lr[1]})

    scheduler.step()
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    msg = f"train epoch:{epoch + 1} / loss:{ret_loss:.4} / SRCC:{rho_s:.4} / PLCC:{rho_p:.4}"
    logging.info(msg)
    print(msg)

    return ret_loss, rho_s, rho_p


def eval_epoch(config, epoch, net, criterion, test_loader, device, mode="FR"):
    with torch.no_grad():
        losses = []
        net.eval()
        pred_epoch = []
        labels_epoch = []
        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data["d_img_org"].to(device)
                labels = torch.squeeze(data["score"].float()).to(device)

                if mode == "FR":
                    x_r = data["r_img_org"].to(device) if "r_img_org" in data else x_d
                else:
                    x_r = x_d

                if config.num_avg_val == 5:
                    x_d = five_point_crop(i, d_img=x_d, config=config)
                    x_r = five_point_crop(i, d_img=x_r, config=config)
                else:
                    x_d = random_crop(d_img=x_d, config=config)
                    x_r = random_crop(d_img=x_r, config=config)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pred += net(x_d, x_r, mode=mode)

            pred /= config.num_avg_val

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            pred_epoch = np.append(pred_epoch, pred.data.cpu().numpy())
            labels_epoch = np.append(labels_epoch, labels.data.cpu().numpy())

        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        ret_loss = np.mean(losses)
        msg = f"Test epoch {mode}:{epoch + 1} ===== loss:{ret_loss:.4} ===== SRCC:{rho_s:.4} ===== PLCC:{rho_p:.4}"
        logging.info(msg)
        print(msg)
        return ret_loss, rho_s, rho_p


if __name__ == "__main__":
    config = get_option()
    print(f"=======training mode: {config.training_mode}")

    cpu_num = 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    device = get_device()
    print(f"=======using device: {device}")

    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU)

    seed = random.randint(0, 9999) if config.random_seed else config.seed
    print(f"---random seed: {config.random_seed}, seed:{seed}")
    setup_seed(seed)

    model_file = os.path.join(config.output_path, config.model_name)
    os.makedirs(model_file, exist_ok=True)

    set_logging(config)
    logging.info(config)
    logging.info(f"seed used for this training {seed}")

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "YOTO"),
        name=config.model_name,
        config=vars(config) if hasattr(config, "__dict__") else {"model": config.model_name},
    )

    # dataloader
    train_loader, val_loader = prepare_dataloader(config)

    # model
    module = importlib.import_module(f"models.{config.network.lower()}")
    net = module.Net(config, device=str(device))
    net = net.to(device)

    num_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"total params: {num_param / 1e6}M")

    criterion = torch.nn.MSELoss()

    optimizer = build_optimizer(net, config)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.eta_min
    )

    best_srocc = {"FR": 0, "NR": 0}
    best_plcc = {"FR": 0, "NR": 0}
    saved_ckpts = {"FR": [], "NR": []}

    for epoch in range(config.n_epoch):
        start_time = time.time()
        logging.info(f"Running training epoch {epoch + 1}")
        loss_val, rho_s, rho_p = train_epoch(
            epoch, net, criterion, optimizer, scheduler,
            train_loader, device, mode=config.training_mode,
            log_interval=config.log_interval,
        )

        log_dict = {"train/loss": loss_val, "train/srcc": rho_s, "train/plcc": rho_p}

        if (epoch + 1) % config.val_freq == 0:
            logging.info("Starting eval...")

            for mode in ("FR", "NR"):
                if config.training_mode not in ("mix", mode):
                    continue

                logging.info(f"Running {mode} testing in epoch {epoch + 1}")
                test_loss, trho_s, trho_p = eval_epoch(
                    config, epoch, net, criterion, val_loader, device, mode=mode
                )

                log_dict.update({
                    f"test_{mode}/loss": test_loss,
                    f"test_{mode}/srcc": trho_s,
                    f"test_{mode}/plcc": trho_p,
                })

                if trho_s > best_srocc[mode] or trho_p > best_plcc[mode]:
                    best_srocc[mode] = max(best_srocc[mode], trho_s)
                    best_plcc[mode] = max(best_plcc[mode], trho_p)
                    ckpt_name = f"{mode}_epoch{epoch + 1}_plcc_{trho_p:.4f}_srocc_{trho_s:.4f}.pth"
                    model_save_path = os.path.join(config.output_path, config.model_name, ckpt_name)
                    torch.save(net.state_dict(), model_save_path)
                    logging.info(f"Saving {mode} weights epoch{epoch + 1}, SRCC:{trho_s}, PLCC:{trho_p}")

                    keep_n = getattr(config, "keep_checkpoints", 0)
                    if keep_n > 0:
                        saved_ckpts[mode].append(model_save_path)
                        while len(saved_ckpts[mode]) > keep_n:
                            old = saved_ckpts[mode].pop(0)
                            if os.path.exists(old):
                                os.remove(old)
                                logging.info(f"Removed old checkpoint: {old}")

            logging.info("Eval done...")

        wandb.log(log_dict)

    wandb.finish()
