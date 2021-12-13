import torch
from torch import nn


class CycleGanLoss(nn.Module):
    def __init__(self, adversarial="True"):
        super(CycleGanLoss, self).__init__()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_id = nn.L1Loss()
        self.adv_loss = None
        self.adversarial = (adversarial == "True")
        if self.adversarial:
            self.adv_loss = AdvLoss()

    def forward(self, id_img, recon_img, real_img, discr_real_out=None, discr_fake_out=None):
        id_loss = self.criterion_id(id_img, real_img)
        cycle_loss = self.criterion_cycle(recon_img, real_img)
        discr_loss, gen_loss = None, None
        if self.adv_loss is not None:
            if discr_real_out is None or discr_fake_out is None:
                raise RuntimeError("Adversarial loss is not None but discriminator outputs are None")
            discr_loss, gen_loss = self.adv_loss(discr_real_out, discr_fake_out)

        return id_loss, cycle_loss, discr_loss, gen_loss


class AdvLoss(nn.Module):
    def __init__(self):
        super(AdvLoss, self).__init__()
        self.criterion_adv = nn.MSELoss()

    def forward(self, discr_real_out, discr_fake_out):
        discr_loss = (self.criterion_adv(discr_real_out, torch.ones(discr_real_out.size())) +
                      self.criterion_adv(discr_fake_out, torch.zeros(discr_fake_out.size()))) * 0.5
        gen_loss = self.criterion_adv(discr_fake_out, torch.ones(discr_fake_out.size()))
        return discr_loss, gen_loss  # discriminator A, generator B
