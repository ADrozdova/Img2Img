import torch
from torch import nn


class CycleGanLoss(nn.Module):
    def __init__(self, types, discr_loss_coef=0.5):
        super(CycleGanLoss, self).__init__()
        self.device = None
        self.criterion_cycle = nn.L1Loss() if "cycle" in types else None
        self.adversarial = "adversarial" in types
        self.adv_loss = AdvLoss(discr_loss_coef) if self.adversarial else None

    def to(self, device):
        self.device = device
        if self.adversarial:
            self.adv_loss = self.adv_loss.to(device)
        return self

    def forward(
        self,
        recon_img,
        real_img,
        discr_real_out=None,
        discr_fake_out=None,
        discr_fake_out_detached=None,
    ):
        cycle_loss = self.criterion_cycle(recon_img, real_img)
        discr_loss, gen_loss = None, None
        if self.adv_loss is not None:
            if (
                discr_real_out is None
                or discr_fake_out is None
                or discr_fake_out_detached is None
            ):
                raise RuntimeError(
                    "Adversarial loss is not None but discriminator outputs are None"
                )
            discr_loss, gen_loss = self.adv_loss(
                discr_real_out, discr_fake_out, discr_fake_out_detached
            )

        return cycle_loss, discr_loss, gen_loss


class AdvLoss(nn.Module):
    def __init__(self, discr_loss_coef=0.5):
        super(AdvLoss, self).__init__()
        self.criterion_adv = nn.MSELoss()
        self.device = None
        self.discr_loss_coef = discr_loss_coef

    def to(self, device):
        self.device = device
        return self

    def forward(self, discr_real_out, discr_fake_out, discr_fake_out_detached):
        ones = torch.ones(discr_real_out.size(), device=self.device)
        zeros = torch.zeros(discr_fake_out.size(), device=self.device)

        discr_loss = (
            self.criterion_adv(discr_real_out, ones)
            + self.criterion_adv(discr_fake_out_detached, zeros)
        ) * self.discr_loss_coef
        gen_loss = self.criterion_adv(discr_fake_out, ones)
        return discr_loss, gen_loss  # discriminator A, generator B
