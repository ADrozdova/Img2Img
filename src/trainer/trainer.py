import random
from random import shuffle
import itertools

import PIL
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
from src.aligner.grapheme_aligner import GraphemeAligner
from src.aligner.aligner import Aligner

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(BaseTrainer):
    def __init__(
            self,
            gen_B,
            gen_A,
            disc_A,
            disc_B,
            criterion,
            optimizer,
            config,
            device,
            data_loader,
            valid_data_loader=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(gen_B, gen_A, disc_A, disc_B, criterion, optimizer, config, device)

        self.gen_B = gen_B
        self.gen_A = gen_A
        self.disc_A = disc_A
        self.disc_B = disc_B

        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 10

        self.train_metrics = MetricTracker(
            "gan loss", "cycle loss", "id loss", "grad norm"
        )
        self.valid_metrics = MetricTracker(
            "gan loss", "cycle loss", "id loss"
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        batch = batch.to(device)
        return batch

    def _train_epoch(self, epoch):
        self.gen_B.train()
        self.gen_A.train()
        self.disc_A.train()
        self.disc_B.train()

        self.train_metrics.reset()
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    for p in itertools.chain(self.gen_B.parameters(), self.gen_A.parameters(),
                                             self.disc_A.parameters(), self.disc_B.parameters()):
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:

                self._log_spectrogram_audio(batch.melspec, batch.melspec_pred, batch.waveform, batch.melspec_pred)
                self._log_scalars(self.train_metrics)
            if batch_idx >= self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):


        # metrics.update("loss", loss.item())
        # metrics.update("mel loss", mel_loss.item())
        # metrics.update("duration loss", duration_loss.item())

        return batch

    def _valid_epoch(self, epoch):
        self.gen_B.eval()
        self.gen_A.eval()
        self.disc_A.eval()
        self.disc_B.eval()

        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(self.valid_data_loader),
                    desc="validation",
                    total=len(self.valid_data_loader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.valid_metrics,
                )

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
