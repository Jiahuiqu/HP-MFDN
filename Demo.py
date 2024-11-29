import os
import argparse
import torch.distributed as dist
import torch.autograd
from torch.distributed.elastic.multiprocessing.errors import record

from Trainer import Trainer

def init_dist_mode(local_rank):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    rank = int(os.environ['RANK'])
    # print(f"rank: {rank}")
    world_size = int(os.environ['WORLD_SIZE'])
    # print(f"world_size: {world_size}")
    local_rank = local_rank
    # print(f"local_rank: {local_rank}")

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    torch.cuda.set_device(local_rank)

    dist.barrier()


def pansharpening(local_rank):
    world_size = int(os.environ['WORLD_SIZE'])
    init_dist_mode(local_rank)
    trainer = Trainer(local_rank, world_size, max_epoch = 1000, lr = 0.0001, batchsize = 1)
    trainer.train()
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    local_rank = args.local_rank

    pansharpening(local_rank)




