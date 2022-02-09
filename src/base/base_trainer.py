import itertools
from abc import abstractmethod

import torch
from numpy import inf

from src.logger import get_visualizer
from src.utils.init_models import init_gen, init_disc


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, criterion, config, device, local_rank, adversarial=True):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.local_rank = local_rank

        self.gen_A, self.gen_B = init_gen(config, device, local_rank)

        self.adversarial = adversarial

        if adversarial:
            self.disc_A, self.disc_B = init_disc(config, device, local_rank)
        else:
            self.disc_A, self.disc_B = None, None

        self.criterion = criterion
        self.criterion = self.criterion.to(self.device)

        trainable_params = filter(
            lambda p: p.requires_grad,
            itertools.chain(self.gen_A.parameters(), self.gen_B.parameters()),
        )
        self.optimizer_G = config.init_obj(
            config["optimizer"], torch.optim, trainable_params
        )

        if adversarial:
            trainable_params = filter(
                lambda p: p.requires_grad, self.disc_A.parameters()
            )
            self.optimizer_DA = config.init_obj(
                config["optimizer"], torch.optim, trainable_params
            )

            trainable_params = filter(
                lambda p: p.requires_grad, self.disc_B.parameters()
            )
            self.optimizer_DB = config.init_obj(
                config["optimizer"], torch.optim, trainable_params
            )
        else:
            self.optimizer_DA, self.optimizer_DB = None, None

        # self.lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

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

        # setup visualization writer instance
        self.writer = get_visualizer(config, self.logger, cfg_trainer["visualize"]) if local_rank == 0 else None

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            if self.local_rank == 0:
                # print logged informations to the screen
                for key, value in log.items():
                    self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        if self.local_rank != 0:
            return

        arch_gen = type(self.gen_A).__name__

        state = {
            "generator": arch_gen,
            "epoch": epoch,
            "state_dict_gen_A": self.gen_A.state_dict(),
            "state_dict_gen_B": self.gen_B.state_dict(),
            "state_dict_disc_A": self.disc_A.state_dict() if self.adversarial else None,
            "state_dict_disc_B": self.disc_B.state_dict() if self.adversarial else None,
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_DA": self.optimizer_DA.state_dict()
            if self.adversarial
            else None,
            "optimizer_DB": self.optimizer_DB.state_dict()
            if self.adversarial
            else None,
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            torch.distributed.barrier()
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        if self.local_rank == 0:
            self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["generator"] != self.config["generator"]:
            if self.local_rank == 0:
                self.logger.warning(
                    "Warning: Architecture configuration given in config file is different from that of "
                    "checkpoint. This may yield an exception while state_dict is being loaded."
                )
        self.gen_A.load_state_dict(
            checkpoint["state_dict_gen_A"]
        )
        self.gen_B.load_state_dict(
            checkpoint["state_dict_gen_B"]
        )

        self.gen_A = torch.nn.parallel.DistributedDataParallel(
            self.gen_A, device_ids=[self.local_rank], output_device=self.local_rank
        )
        self.gen_B = torch.nn.parallel.DistributedDataParallel(
            self.gen_B, device_ids=[self.local_rank], output_device=self.local_rank
        )
        if checkpoint["config"]["discriminator"] != self.config["discriminator"]:
            if self.local_rank == 0:
                self.logger.warning(
                    "Warning: Architecture configuration given in config file is different from that of "
                    "checkpoint. This may yield an exception while state_dict is being loaded."
                )

        if self.adversarial:
            self.disc_A.load_state_dict(
                checkpoint["state_dict_disc_A"]
            )
            self.disc_B.load_state_dict(
                checkpoint["state_dict_disc_B"]
            )

            self.disc_A = torch.nn.parallel.DistributedDataParallel(
                self.disc_A, device_ids=[self.local_rank], output_device=self.local_rank
            )
            self.disc_B = torch.nn.parallel.DistributedDataParallel(
                self.disc_B, device_ids=[self.local_rank], output_device=self.local_rank
            )

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
        ):
            if self.local_rank == 0:
                self.logger.warning(
                    "Warning: Optimizer or lr_scheduler given in config file is different "
                    "from that of checkpoint. Optimizer parameters not being resumed."
                )
        else:
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            if self.adversarial:
                self.optimizer_DA.load_state_dict(checkpoint["optimizer_DA"])
                self.optimizer_DB.load_state_dict(checkpoint["optimizer_DB"])
        if self.local_rank == 0:
            self.logger.info(
                "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
            )
