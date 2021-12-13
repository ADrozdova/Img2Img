from abc import abstractmethod

import torch
from numpy import inf

from src.base import BaseModel


class BaseTrainer:
    def __init__(self, gen_B: BaseModel, gen_A: BaseModel, discr_A: BaseModel, discr_B: BaseModel,
                 criterion, optimizer, config, device):
        self.device = device
        self.config = config

        self.gen_B = gen_B
        self.gen_A = gen_A
        self.discr_A = discr_A
        self.discr_B = discr_B

        self.criterion = criterion
        self.optimizer = optimizer

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            log = {"epoch": epoch}
            log.update(result)

            best = False
            if self.mnt_mode != "off":
                try:
                    improved = (
                                   self.mnt_mode == "min" and log[
                                   self.mnt_metric] <= self.mnt_best
                               ) or (
                                   self.mnt_mode == "max" and log[
                                   self.mnt_metric] >= self.mnt_best
                               )
                except KeyError:
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        arch_g = type(self.gen_B).__name__
        arch_d = type(self.discr_A).__name__

        state = {
            "arch_g": arch_g,
            "arch_d": arch_d,
            "epoch": epoch,
            "state_dict_gen_a": self.gen_A.state_dict(),
            "state_dict_gen_b": self.gen_B.state_dict(),
            "state_dict_discr_a": self.discr_A.state_dict(),
            "state_dict_disrc_b": self.discr_B.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
