import src.model as module_arch
import torch


def init_gen(config, device, local_rank):
    logger = config.get_logger("train")
    gen_B = config.init_obj(config["generator"], module_arch)
    gen_A = config.init_obj(config["generator"], module_arch)

    logger.info(gen_A)

    gen_B = gen_B.to(device)
    gen_A = gen_A.to(device)

    gen_A = torch.nn.parallel.DistributedDataParallel(
        gen_A, device_ids=[local_rank], output_device=local_rank
    )
    gen_B = torch.nn.parallel.DistributedDataParallel(
        gen_B, device_ids=[local_rank], output_device=local_rank
    )

    return gen_A, gen_B


def init_disc(config, device, local_rank):
    logger = config.get_logger("train")
    disc_A = config.init_obj(config["discriminator"], module_arch)
    disc_B = config.init_obj(config["discriminator"], module_arch)

    logger.info(disc_A)

    disc_A = disc_A.to(device)
    disc_B = disc_B.to(device)

    disc_A = torch.nn.parallel.DistributedDataParallel(
        disc_A, device_ids=[local_rank], output_device=local_rank
    )
    disc_B = torch.nn.parallel.DistributedDataParallel(
        disc_B, device_ids=[local_rank], output_device=local_rank
    )

    return disc_A, disc_B
