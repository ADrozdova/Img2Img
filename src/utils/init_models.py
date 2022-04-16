import src.model as module_arch
import torch


def init_gen(config, device, device_ids):
    logger = config.get_logger("train")
    gen_B = config.init_obj(config["generator"], module_arch)
    gen_A = config.init_obj(config["generator"], module_arch)

    logger.info(gen_A)

    gen_B = gen_B.to(device)
    gen_A = gen_A.to(device)

    if len(device_ids) > 1:
        gen_B = torch.nn.DataParallel(gen_B, device_ids=device_ids)
        gen_A = torch.nn.DataParallel(gen_A, device_ids=device_ids)

    return gen_A, gen_B


def init_disc(config, device, device_ids):
    logger = config.get_logger("train")
    disc_A = config.init_obj(config["discriminator"], module_arch)
    disc_B = config.init_obj(config["discriminator"], module_arch)

    logger.info(disc_A)

    disc_A = disc_A.to(device)
    disc_B = disc_B.to(device)

    if len(device_ids) > 1:
        disc_A = torch.nn.DataParallel(disc_A, device_ids=device_ids)
        disc_B = torch.nn.DataParallel(disc_B, device_ids=device_ids)

    return disc_A, disc_B
