import os
import sys
import random

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from data import CustomDataParallel
from data.dataload import CustomDataset
from models.tcm import TCM
from utils.loss import RateDistortionLoss, AverageMeter
from utils.in_out import parse_args, save_checkpoint
from utils.optimize import configure_optimizers


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, type="mse"):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 1000 == 0:
            if type == "mse":
                print(
                    f"Train epoch {epoch}:["
                    f"Batch: {i}/{len(train_dataloader)} -- {100 * i / len(train_dataloader): .0f}%"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSELoss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBppLoss: {out_criterion["bpp_loss"].item():.3f} |'
                    f'\tAuxLoss: {aux_loss.item():.2f}]'
                )
            else:
                print(
                    f"Train epoch {epoch}:["
                    f"Batch: {i}/{len(train_dataloader)} -- {100 * i / len(train_dataloader): .0f}%"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS-SSIMLoss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBppLoss: {out_criterion["bpp_loss"].item():.3f} |'
                    f'\tAuxLoss: {aux_loss.item():.2f}]'
                )

    return


def test_one_epoch(epoch, test_dataloader, model, criterion, type="mse"):
    model.eval()
    device = next(model.parameters()).device
    loss = None  # Init loss
    if type == "mse":
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
            print(
                f"Test epoch {epoch} - Average losses:"
                f'\tLoss: {loss.avg:.3f} |'
                f'\tMSELoss: {mse_loss.avg:.3f} |'
                f'\tBppLoss: {bpp_loss.avg:.3f} |'
                f'\tAuxLoss: {aux_loss.avg:.2f}'
            )

    if type == "ms-ssim":
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])
            print(
                f"Test epoch {epoch} - Average losses:"
                f'\tLoss: {loss.avg:.3f} |'
                f'\tMSELoss: {ms_ssim_loss.avg:.3f} |'
                f'\tBppLoss: {bpp_loss.avg:.3f} |'
                f'\tAuxLoss: {aux_loss.avg:.2f}'
            )

    return loss.avg


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, "tensorboard"))
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    writer = SummaryWriter(os.path.join(save_path, "tensorboard"))

    train_dataset = CustomDataset(args, mode="train", transform=True)
    test_dataset = CustomDataset(args, mode="test", transform=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda")
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda")
    )

    model = TCM()
    model.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)

    optimizer, aux_optimizer = configure_optimizers(model, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones, 0.1, -1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    # Training
    last_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint} ...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        if args.contune_train:
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_schedule.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("-inf")
    for epoch in range(last_epoch, args.epochs):
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, type)
        loss = test_one_epoch(epoch, test_dataloader, model, criterion, type)
        writer.add_scalar("test_loss", loss, epoch)
        lr_schedule.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizzer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_schedule.state_dict()
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
