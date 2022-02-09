import io
import itertools

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.utils import inf_loop, MetricTracker
from torch.nn.utils import clip_grad_norm_


class Trainer(BaseTrainer):
    def __init__(
        self,
        criterion,
        config,
        device,
        local_rank,
        data_loader_A,
        data_loader_B,
        valid_data_loader_A=None,
        valid_data_loader_B=None,
        gen_scheduler=None,
        disc_scheduler=None,
        skip_oom=True,
        adversarial=True,
    ):
        super().__init__(
            criterion,
            config,
            device,
            local_rank,
            adversarial,
        )
        self.skip_oom = skip_oom
        self.config = config

        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.valid_data_loader_A = valid_data_loader_A
        self.valid_data_loader_B = valid_data_loader_B

        self.len_epoch = len(self.data_loader_A)

        self.do_validation = self.valid_data_loader_A is not None
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.log_step = config["trainer"]["log_step"]
        self.start_epoch = 1

        self.epochs = config["trainer"]["epochs"]
        self.save_period = config["trainer"]["save_period"]
        self.checkpoint_dir = config.save_dir

        self.train_metrics = MetricTracker(
            "generator_loss",
            "disc_A_loss",
            "disc_B_loss",
            "grad_norm",
            writer=self.writer,
        )
        self.valid_metrics = MetricTracker(
            "generator_loss", "disc_A_loss", "disc_B_loss", writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        batch = batch.to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.gen_A.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            clip_grad_norm_(
                self.gen_B.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            if self.adversarial:
                clip_grad_norm_(
                    self.disc_A.parameters(), self.config["trainer"]["grad_norm_clip"]
                )
                clip_grad_norm_(
                    self.disc_B.parameters(), self.config["trainer"]["grad_norm_clip"]
                )

    def _train_epoch(self, epoch):
        self.gen_B.train()
        self.gen_A.train()
        if self.disc_A is not None:
            self.disc_A.train()
            self.disc_B.train()

        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        gen_loss, discr_A_loss, discr_B_loss = [], [], []

        for batch_idx, batch in enumerate(
            tqdm(
                zip(self.data_loader_A, self.data_loader_B),
                desc="train",
                total=self.len_epoch,
            )
        ):

            gen_loss_i, discr_A_loss_i, discr_B_loss_i = self.process_batch(
                batch,
                is_train=True,
                metrics=self.train_metrics,
                log=(batch_idx % 10 == 0),
            )
            gen_loss.append(gen_loss_i)
            discr_A_loss.append(discr_A_loss_i)
            discr_B_loss.append(discr_B_loss_i)

            self.writer.add_scalar("generator_loss_train", gen_loss_i)
            self.writer.add_scalar("disc_A_loss_train", discr_A_loss_i)
            self.writer.add_scalar("disc_B_loss_train", discr_B_loss_i)

            if batch_idx >= self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.disc_scheduler is not None:
            self.disc_scheduler.step()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, log=False):
        real_A, real_B = batch

        real_A = self.move_batch_to_device(real_A, self.device)
        real_B = self.move_batch_to_device(real_B, self.device)

        fake_B = self.gen_B(real_A)
        fake_A = self.gen_A(real_B)

        name = "train" if is_train else "valid"

        if log:
            self._log_img("real_A_" + name, real_A)
            self._log_img("real_B_" + name, real_B)
            self._log_img("fake_A_" + name, fake_A)
            self._log_img("fake_B_" + name, fake_B)

        if self.adversarial:
            disc_real_A = self.disc_A(real_A)
            disc_fake_A = self.disc_A(fake_A)

            fake_A_detached = fake_A.clone().detach()
            disc_fake_A_detached = self.disc_A(fake_A_detached)
        else:
            (
                disc_real_A,
                disc_fake_A,
                disc_fake_A_detached,
            ) = (None, None, None)

        id_A = self.gen_A(real_A)
        recon_A = self.gen_A(fake_B)

        id_A_loss, cycle_A_loss, discr_A_loss, gen_B_loss = self.criterion(
            id_A, recon_A, real_A, disc_real_A, disc_fake_A, disc_fake_A_detached
        )

        id_B = self.gen_B(real_B)
        recon_B = self.gen_B(fake_A)

        if self.adversarial:
            disc_real_B = self.disc_A(real_B)
            disc_fake_B = self.disc_A(fake_B)
            fake_B_detached = fake_B.clone().detach()
            disc_fake_B_detached = self.disc_A(fake_B_detached)

        else:
            (
                disc_real_B,
                disc_fake_B,
                disc_fake_B_detached,
            ) = (None, None, None)

        id_B_loss, cycle_B_loss, discr_B_loss, gen_A_loss = self.criterion(
            id_B, recon_B, real_B, disc_real_B, disc_fake_B, disc_fake_B_detached
        )

        gen_loss = (
            self.config["loss"]["lambda_id"] * (id_A_loss + id_B_loss)
            + self.config["loss"]["lambda_cyc"] * (cycle_A_loss + cycle_B_loss)
        ) * 0.5

        if self.adversarial:
            gen_loss += (gen_A_loss + gen_B_loss) * 0.5

        if is_train:
            self.optimizer_G.zero_grad()
            gen_loss.backward()
            self.optimizer_G.step()

            if self.adversarial:
                self.optimizer_DA.zero_grad()
                self.optimizer_DB.zero_grad()

                discr_A_loss.backward()
                discr_B_loss.backward()

                self.optimizer_DA.step()
                self.optimizer_DB.step()

        metrics.update("generator_loss", gen_loss.item())
        metrics.update("disc_A_loss", discr_A_loss.item())
        metrics.update("disc_B_loss", discr_B_loss.item())

        if self.adversarial:
            return gen_loss.item(), discr_A_loss.item(), discr_B_loss.item()
        else:
            return gen_loss.item(), None, None

    def _valid_epoch(self, epoch):
        self.gen_B.eval()
        self.gen_A.eval()
        if self.adversarial:
            self.disc_A.eval()
            self.disc_B.eval()

        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(
                    zip(self.valid_data_loader_A, self.valid_data_loader_B),
                    desc="valid",
                    total=len(self.valid_data_loader_A),
                )
            ):
                log = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.valid_metrics,
                    log=(batch_idx % 10 == 0),
                )
                self.writer.set_step(epoch * self.len_epoch, "valid")
                self._log_scalars(self.valid_metrics)

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader_A, "n_samples"):
            current = batch_idx * self.data_loader_A.batch_size
            total = self.data_loader_A.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _log_img(self, name, image):
        img = image[0].permute(1, 2, 0).detach().cpu()
        img = PIL.Image.open(self._plot_img_to_buf(img))
        self.writer.add_image(name, ToTensor()(img))

    def _plot_img_to_buf(self, img_tensor, name=None):
        plt.figure(figsize=(20, 20))
        plt.imshow((img_tensor.numpy() * 255).astype("uint8"))
        plt.title(name)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return buf
