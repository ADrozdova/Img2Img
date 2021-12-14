import itertools

import numpy as np
import torch
from tqdm import tqdm


class Trainer():
    def __init__(
            self,
            gen_B,
            gen_A,
            disc_A,
            disc_B,
            criterion,
            optimizer_G,
            optimizer_DA,
            optimizer_DB,
            config,
            device,
            data_loader_A,
            data_loader_B,
            valid_data_loader_A=None,
            valid_data_loader_B=None,
            lr_scheduler=None,
            skip_oom=True,
    ):
        self.gen_B = gen_B
        self.gen_A = gen_A
        self.disc_A = disc_A
        self.disc_B = disc_B

        self.device = device
        self.config = config

        self.criterion = criterion
        self.criterion = self.criterion.to(self.device)
        self.optimizer_G = optimizer_G
        self.optimizer_DA = optimizer_DA
        self.optimizer_DB = optimizer_DB

        self.skip_oom = skip_oom
        self.config = config

        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.valid_data_loader_A = valid_data_loader_A
        self.valid_data_loader_B = valid_data_loader_B

        self.len_epoch = len(self.data_loader_A)

        self.do_validation = self.valid_data_loader_A is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 10
        self.start_epoch = 1

        self.epochs = config["trainer"]["epochs"]
        self.save_period = config["trainer"]["save_period"]
        self.checkpoint_dir = config.save_dir


    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        batch = batch.to(device)
        return batch

    def _train_epoch(self, epoch):
        self.gen_B.train()
        self.gen_A.train()
        self.disc_A.train()
        self.disc_B.train()

        gen_loss, discr_A_loss, discr_B_loss = [], [], []

        for batch_idx, batch in enumerate(
                tqdm(zip(self.data_loader_A, self.data_loader_B), desc="train", total=self.len_epoch)
        ):
            try:
                gen_loss_i, discr_A_loss_i, discr_B_loss_i = self.process_batch(
                    batch,
                    is_train=True,
                )
                gen_loss.append(gen_loss_i)
                discr_A_loss.append(discr_A_loss_i)
                discr_B_loss.append(discr_B_loss_i)
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

            if batch_idx >= self.len_epoch:
                break
        return np.mean(gen_loss), np.mean(discr_A_loss), np.mean(discr_B_loss)

    def process_batch(self, batch, is_train: bool):
        real_A, real_B = batch
        real_A = self.move_batch_to_device(real_A, self.device)
        real_B = self.move_batch_to_device(real_B, self.device)

        fake_B = self.gen_B(real_A)
        recon_A = self.gen_A(fake_B)

        fake_A = self.gen_A(real_B)
        recon_B = self.gen_B(fake_A)

        id_A = self.gen_A(real_A)
        id_B = self.gen_B(real_B)

        if self.criterion.adversarial:
            disc_real_A = self.disc_A(real_A)
            disc_fake_A = self.disc_A(fake_A)

            fake_A_detached = fake_A.clone().detach()
            disc_fake_A_detached = self.disc_A(fake_A_detached)

            disc_real_B = self.disc_A(real_B)
            disc_fake_B = self.disc_A(fake_B)

            fake_B_detached = fake_B.clone().detach()
            disc_fake_B_detached = self.disc_A(fake_B_detached)
        else:
            disc_real_A, disc_fake_A, disc_real_B, disc_fake_B, disc_fake_A_detached, disc_fake_B_detached =\
                None, None, None, None, None, None

        id_A_loss, cycle_A_loss, discr_A_loss, gen_B_loss = self.criterion(id_A, recon_A, real_A, disc_real_A,
                                                                           disc_fake_A, disc_fake_A_detached)
        id_B_loss, cycle_B_loss, discr_B_loss, gen_A_loss = self.criterion(id_B, recon_B, real_B, disc_real_B,
                                                                           disc_fake_B, disc_fake_B_detached)

        gen_loss = (gen_A_loss + gen_B_loss + self.config["loss"]["lambda_id"] * (id_A_loss + id_B_loss) +
                    self.config["loss"]["lambda_cyc"] * (cycle_A_loss + cycle_B_loss)) * 0.5

        if is_train:
            self.optimizer_G.zero_grad()
            gen_loss.backward()
            self.optimizer_G.step()

            if self.criterion.adversarial:
                self.optimizer_DA.zero_grad()
                self.optimizer_DB.zero_grad()

                discr_A_loss.backward()
                discr_B_loss.backward()

                self.optimizer_DA.step()
                self.optimizer_DB.step()

        return gen_loss.item(), discr_A_loss.item(), discr_B_loss.item()

    def _valid_epoch(self, epoch):
        self.gen_B.eval()
        self.gen_A.eval()
        self.disc_A.eval()
        self.disc_B.eval()

        gen_loss, discr_A_loss, discr_B_loss = [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                    tqdm(zip(self.valid_data_loader_A, self.valid_data_loader_A), desc="train",
                         total=len(self.valid_data_loader_A))
            ):
                gen_loss_i, discr_A_loss_i, discr_B_loss_i = self.process_batch(
                    batch,
                    is_train=False,
                )
                gen_loss.append(gen_loss_i)
                discr_A_loss.append(discr_A_loss_i)
                discr_B_loss.append(discr_B_loss_i)

        return np.mean(gen_loss), np.mean(discr_A_loss), np.mean(discr_B_loss)

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        gen_loss, discr_A_loss, discr_B_loss = [], [], []

        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            gen_loss_i, discr_A_loss_i, discr_B_loss_i = self._train_epoch(epoch)

            gen_loss.append(gen_loss_i)
            discr_A_loss.append(discr_A_loss_i)
            discr_B_loss.append(discr_B_loss_i)

            if (epoch + 1) % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        arch_g = type(self.gen_B).__name__
        arch_d = type(self.disc_A).__name__

        state = {
            "arch_g": arch_g,
            "arch_d": arch_d,
            "epoch": epoch,
            "state_dict_gen_a": self.gen_A.state_dict(),
            "state_dict_gen_b": self.gen_B.state_dict(),
            "state_dict_discr_a": self.disc_A.state_dict(),
            "state_dict_disrc_b": self.disc_B.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_DA": self.optimizer_DA.state_dict(),
            "optimizer_DB": self.optimizer_DB.state_dict(),
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
