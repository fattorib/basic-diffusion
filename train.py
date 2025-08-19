import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
from torchvision.utils import make_grid
from einops import rearrange
from time import time
from math import sqrt, log
from unet import UNetConfig, UNet
from diffusion import DiffusionWrapper, EWMAWrapper
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cycle(loader):
    while True:
        for data in loader:
            yield data


class NormalizeTransform(torch.nn.Module):
    # scales the data from [0,1] to [-1,1]
    def forward(self, img):
        return (img * 2.0) - 1.0


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def evaluate_bpd(
    loader,
    ddpm: DiffusionWrapper,
    steps: int,
    subsample: int | None,
) -> float:
    npd_loss = 0.0

    for vstep in tqdm(
        range(steps),
        total=steps,
        desc="Computing NLL VLB (bits/dim)",
    ):
        batch, _ = next(loader)

        if vstep >= steps:
            break

        batch = batch.to("cuda", non_blocking=True)

        if subsample is not None:
            npd_loss += ddpm.compute_vlb_npd_strided(batch, subsample)

        else:
            npd_loss += ddpm.compute_vlb_npd(batch)

    return npd_loss / (steps * log(2.0))


def save_checkpoint(model: torch.nn.Module, step: int, config: dict, filename: str):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "config": config,
    }
    torch.save(checkpoint, filename)


def prepare_grid(
    samples: torch.Tensor, dims: tuple[int, int, int], n_samples: int
) -> np.ndarray:
    C, H, W = dims
    samples = samples.reshape(-1, C, H, W)

    samples = unnormalize(samples)

    grid = make_grid(samples, nrow=int(sqrt(n_samples)))
    return grid.permute(1, 2, 0).cpu().detach().numpy()


def generate_samples(
    n_samples: int,
    ddpm: DiffusionWrapper,
    dims: tuple[int, int, int],
    subsample: int | None,
) -> np.ndarray:
    samples = ddpm.generate_samples(n_samples=n_samples, subsample=subsample)

    return prepare_grid(samples, dims, n_samples)


def create_dataset(dataset: str):
    match dataset:
        case "CIFAR10":
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    NormalizeTransform(),
                ]
            )
            valid_transform = transforms.Compose(
                [transforms.ToTensor(), NormalizeTransform()]
            )

            train_dataset = CIFAR10(
                root="./data",
                train=True,
                transform=train_transform,
                download=True,
            )

            val_dataset = CIFAR10(
                root="./data",
                train=False,
                transform=valid_transform,
                download=True,
            )
            in_hw = 32

        case _:
            raise NotImplementedError

    return in_hw, train_dataset, val_dataset


def main():
    torch.manual_seed(3601627)
    max_t = 1000
    max_lr = 1e-4

    val_sample_freq = 10000
    val_steps = 25

    nll_freq = 10000
    nll_steps = 5

    n_samples = 256
    schedule_type = "cosine"
    n_train_steps = 200001
    batch_size = 128
    gas = 1

    in_chann = 3

    generate = True
    compute_bpd = True
    compile = True

    dataset = "CIFAR10"
    assert schedule_type in ["linear", "cosine"]

    in_hw, train_dataset, val_dataset = create_dataset(dataset)

    subsample_steps = [100, 50]

    model = UNet(
        UNetConfig(
            3,
            base_hidden=128,
            scales=[1, 2, 2, 2],  # 32M param
            n_heads=4,
            p_drop=0.3,
            attn_resolutions=[16, 8],
            in_resolution=in_hw,
            pred_variance=True,
            n_resblocks=2,
        )
    )

    config_dict = model.config.as_dict()
    config_dict["dataset"] = dataset
    config_dict["hparams.max_t"] = max_t
    config_dict["hparams.max_lr"] = max_lr
    config_dict["hparams.val_sample_freq"] = val_sample_freq
    config_dict["hparams.val_steps"] = val_steps
    config_dict["hparams.nll_freq"] = nll_freq
    config_dict["hparams.nll_steps"] = nll_steps
    config_dict["hparams.n_samples"] = n_samples
    config_dict["hparams.schedule_type"] = schedule_type
    config_dict["hparams.n_train_steps"] = n_train_steps
    config_dict["hparams.batch_size"] = batch_size
    config_dict["hparams.model_size"] = sum(p.numel() for p in model.parameters())

    logger.info(f"{config_dict=}")
    model.cuda()

    if compile:
        model: UNet = torch.compile(model)  # type: ignore

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
    )

    ddpm = DiffusionWrapper(
        model, schedule_type=schedule_type, max_t=max_t, in_dim=in_hw
    )

    ema_ddpm = DiffusionWrapper(
        EWMAWrapper(model, beta=0.9999, compile=compile),
        schedule_type=schedule_type,
        max_t=max_t,
        in_dim=in_hw,
    )

    assert isinstance(ema_ddpm.model, EWMAWrapper)
    assert isinstance(ddpm.model, torch.nn.Module)

    _ = wandb.init(resume="allow", project="diffusion-exps")
    wandb.config.update(config_dict)

    train_loader = cycle(
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    )

    val_loader = cycle(
        DataLoader(
            val_dataset,
            batch_size=2 * batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
    )

    for step in tqdm(range(n_train_steps), total=n_train_steps):
        batch, _ = next(train_loader)

        model.train()
        if step >= n_train_steps:
            break

        loss_dict = {}

        batch = batch.to("cuda", non_blocking=True)

        batch = rearrange(batch, "(g b) c h w -> g b c h w", g=gas)

        t0 = time()

        running_loss = 0.0

        for mbatch in batch:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = ddpm.forward_batch(mbatch)
                loss /= gas
            loss.backward()

            running_loss += loss.item()

        t1 = time()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ema_ddpm.model.update(ddpm.model)

        loss_dict["train/loss"] = running_loss
        loss_dict["train/step_time"] = t1 - t0

        if step % val_sample_freq == 0:
            model.eval()

            pred_var_str = "pred_var" if model.config.pred_variance else "no_pred_var"
            checkpoint_filename = (
                f"checkpoint_{schedule_type}_{dataset}_{pred_var_str}.pt"
            )
            save_checkpoint(
                ema_ddpm.model.model, step, config_dict, checkpoint_filename
            )

            if generate:
                # generations from N(0,1)
                grid = generate_samples(
                    n_samples, ema_ddpm, (in_chann, in_hw, in_hw), subsample=None
                )

                images = wandb.Image(grid)  # type: ignore

                loss_dict[f"images/images_{max_t}"] = images

                for subsample in subsample_steps:
                    grid = generate_samples(
                        n_samples,
                        ema_ddpm,
                        (in_chann, in_hw, in_hw),
                        subsample=subsample,
                    )

                    images = wandb.Image(grid)  # type: ignore

                    loss_dict[f"images/images_{subsample}"] = images

            running_loss = 0.0

            for vstep in tqdm(range(val_steps), total=val_steps):
                batch, _ = next(val_loader)

                if vstep >= val_steps:
                    break

                batch = batch.to("cuda", non_blocking=True)

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = ema_ddpm.forward_batch(batch)
                        running_loss += loss.item()

            loss_dict["valid/loss"] = running_loss / val_steps

            if (step % nll_freq == 0) and compute_bpd:
                loss_dict["valid/nll_bpd"] = evaluate_bpd(
                    val_loader, ema_ddpm, nll_steps, None
                )

                for subsample in subsample_steps:
                    loss_dict[f"valid/nll_bpd_{subsample}"] = evaluate_bpd(
                        val_loader, ema_ddpm, nll_steps, subsample
                    )

        wandb.log(loss_dict)  # type: ignore

    pred_var_str = "pred_var" if model.config.pred_variance else "no_pred_var"
    final_filename = f"checkpoint_final_{schedule_type}_{dataset}_{pred_var_str}.pt"
    save_checkpoint(ema_ddpm.model.model, -1, config_dict, final_filename)

    loss_dict = {}

    full_nll_steps = len(val_dataset) // batch_size
    loss_dict["valid/nll_bpd"] = evaluate_bpd(
        val_loader, ema_ddpm, steps=full_nll_steps, subsample=None
    )
    for subsample in subsample_steps:
        loss_dict[f"valid/nll_bpd_{subsample}"] = evaluate_bpd(
            val_loader, ema_ddpm, steps=full_nll_steps, subsample=subsample
        )

    wandb.log(loss_dict)


if __name__ == "__main__":
    main()
