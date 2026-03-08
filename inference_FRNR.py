import os
import torch
import numpy as np
import random
import importlib
from utils.process import five_point_crop, random_crop
from dataloader import prepare_dataloader
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from options import get_option


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
        torch.backends.cudnn.deterministic = True


def eval_epoch(config, net, test_loader, device):
    with torch.no_grad():
        net.eval()
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            x_d = data["d_img_org"].to(device)
            x_r = data["r_img_org"].to(device)
            labels = torch.squeeze(data["score"].float()).to(device)

            for i in range(config.num_avg_val):
                if config.num_avg_val == 5:
                    x_d_crop = five_point_crop(i, d_img=x_d, config=config)
                    x_r_crop = five_point_crop(i, d_img=x_r, config=config)
                else:
                    x_d_crop = random_crop(d_img=x_d, config=config)
                    x_r_crop = random_crop(d_img=x_r, config=config)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    if config.infer_mode == "FR":
                        pred += net(x_d_crop, x_r_crop)
                    elif config.infer_mode == "NR":
                        pred += net(x_d_crop, x_d_crop)
                    else:
                        raise NotImplementedError(f"infer mode '{config.infer_mode}' not implemented")

            pred /= config.num_avg_val

            pred_epoch = np.append(pred_epoch, pred.data.cpu().numpy())
            labels_epoch = np.append(labels_epoch, labels.data.cpu().numpy())

        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        print(f"Test result: ===== SRCC:{rho_s:.4} ===== PLCC:{rho_p:.4}")


if __name__ == "__main__":
    config = get_option()
    print(f"=======inference mode: {config.infer_mode}")

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

    setup_seed(20)

    _, test_loader = prepare_dataloader(config, cross_check=config.cross_check)

    module = importlib.import_module(f"models.{config.network.lower()}")
    net = module.Net(config, device=str(device))
    net.load_state_dict(torch.load(config.checkpoint, map_location=device))
    net = net.to(device)

    eval_epoch(config, net, test_loader, device)
