from Unet import Reconstruction
from torch import optim
from Dataset import HS_Dataload
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.distributed as dist
from scipy.io import savemat
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
def adjust_learning_rate(lr, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False

def calculate_global_accuracy_and_loss(loss, world_size):
    # Converts the count and total of the local GPU to tensor
    local_metrics = torch.tensor([loss], dtype=torch.float64, device='cuda')
    # Wait for all GPUs to finish before calculating the global metrics
    dist.barrier()
    # Sum the count and total of all GPUs together
    dist.all_reduce(local_metrics, op=torch.distributed.ReduceOp.SUM)
    # Calculate the global accuracy and loss
    all_loss = local_metrics / world_size
    return all_loss


def train_epoch(epoch, model, optimizer, criteron, train_loader, show_interview=3):
    model.train()
    loss_meter, count_it = 0, 0
    for step, (pan_data, lr_hs_data, gt_hs_data) in enumerate(train_loader):
        PAN = pan_data.type(torch.float32).cuda()
        LRHS = lr_hs_data.type(torch.float32).cuda()
        gtHS = gt_hs_data.type(torch.float32).cuda()
        out, pan, hs, adv_loss = model(PAN, LRHS)
        loss = criteron(out, gtHS) + criteron(pan, PAN) + criteron(hs, LRHS) + adv_loss.item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        loss_meter += loss
        count_it += 1
        if step % show_interview == 0:
            print("train-----epoch:", epoch, "step:", step + 1, "loss:", loss.item())

    train_loss = calculate_global_accuracy_and_loss(float(loss_meter / count_it), dist.get_world_size())

    return train_loss


def val_epoch(epoch, model, criteron, val_loader, show_interview=3):
    model.eval()
    loss_meter, count_it = 0, 0
    with torch.no_grad():
        for step, (pan_data, lr_hs_data, gt_hs_data) in enumerate(val_loader):
            PAN = pan_data.type(torch.float32).cuda()
            LRHS = lr_hs_data.type(torch.float32).cuda()
            gtHS = gt_hs_data.type(torch.float32).cuda()
            out, pan, hs, adv_loss = model(PAN, LRHS)
            loss = criteron(out, gtHS) + criteron(pan, PAN) + criteron(hs, LRHS) + adv_loss.item()
            loss_meter += loss
            count_it += 1
            if step % show_interview == 0:
                print("#val-----epoch:", epoch, "step:", step + 1, "loss:", loss.item())
        val_loss = calculate_global_accuracy_and_loss(float(loss_meter / count_it), dist.get_world_size())
    return val_loss

def test(model):
    model.eval()
    checkpoint = torch.load('best.mdl')
    print("min_loss:", checkpoint['best_val'])
    print(checkpoint['epoch'])
    model.load_state_dict(checkpoint['state_dict'])
    db_test = HS_Dataload('data', "test", size=160)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False)
    with torch.no_grad():
        for step, (pan_data, lr_hs_data, gt_hs_data) in enumerate(test_loader):
            PAN = pan_data.type(torch.float32).cuda()
            LRHS = lr_hs_data.type(torch.float32).cuda()
            gtHS = gt_hs_data.type(torch.float32).cuda()
            out, pan, hs, _ = model(PAN, LRHS)
            filename = "sub//{0}.mat".format(str("out_" + "{0}").format(step + 1))
            savemat(filename, {"data": out.detach().cpu().numpy()})
        print("save success!!!!")



class Trainer:
    def __init__(self, local_rank, world_size, max_epoch = 1000, lr = 0.0003, batchsize = 4):
        self.local_rank = local_rank
        self.world_size = world_size
        self.max_epoch = max_epoch
        self.batchsize = batchsize
        self.lr = lr
        self.model = Reconstruction().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)

    def train(self):
        set_seed(1)
        db_train = HS_Dataload('data', mode="train", size=128)
        train_sampler = torch.utils.data.distributed.DistributedSampler(db_train)
        train_dataloader = DataLoader(db_train, batch_size=self.batchsize, sampler=train_sampler)
        db_val = HS_Dataload('data', "val", size=128)
        val_sampler = torch.utils.data.distributed.DistributedSampler(db_val, shuffle=False)
        val_dataloader = DataLoader(db_val, batch_size=self.batchsize, sampler=val_sampler)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        self.criteron = nn.L1Loss().cuda()
        if self.local_rank == 0:
            print("****************Start Training****************\n")
        best_loss = 10
        train_loss_list = []
        decay_stage = [300, 500, 700, 950]

        for epoch in range(self.max_epoch):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            train_loss = train_epoch(epoch, self.model, self.optimizer, self.criteron, train_dataloader)
            val_loss = val_epoch(epoch, self.model, self.criteron, val_dataloader)
            train_loss_list.append(train_loss)
            if epoch % 1 == 0:
                print("#epoch:%02d best_loss:%0.7f train_loss:%.7f val_loss:%.7f" % (
                epoch, best_loss, train_loss, val_loss))
            if val_loss <= best_loss:
                state = dict(epoch=epoch + 1, state_dict=self.model.state_dict(), best_val=val_loss)
                torch.save(state, "best.mdl")
                best_loss = val_loss
            if (epoch + 1) in decay_stage:
                self.lr *= 0.3
                adjust_learning_rate(self.lr, self.optimizer)
            if (epoch + 1) % 100 == 0:
                state = dict(epoch=epoch + 1, state_dict=self.model.state_dict(), best_val=val_loss)
                torch.save(state, "best_{0}.mdl".format(epoch + 1))
            torch.cuda.synchronize()