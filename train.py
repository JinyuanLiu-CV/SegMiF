import argparse
import datetime
import logging
import os
import random


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.model_fusion import  Mean, Network3, Fusion_Network3_ac

from datasets import voc_fusion3 as voc
from datasets import voc_fusion2 as voc2

from utils import eval_seg
from utils.optimizer import PolyWarmupAdamW, PolyWarmupAdamW_seg
from omegaconf import OmegaConf
from val_performance import val_fusion, val_segformer
from val_performance import val_fusion_train

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--configf",
                    default='configs/voc_fusion.yaml',
                    type=str,
                    help="configf")
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s') 
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)



def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    #time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def validate(model=None, criterion=None, data_loader=None):


    val_loss = 0.0
    preds, gts = [], []
    model.eval()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            _, inputs, labels = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)

            outputs = model(inputs)
            labels = labels.long().to(outputs.device)

            resized_outputs = F.interpolate(outputs,
                                            size=labels.shape[1:],
                                            mode='bilinear',
                                            align_corners=False)

            loss = criterion(resized_outputs, labels)
            val_loss += loss

            preds += list(
                torch.argmax(resized_outputs,
                             dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    score = eval_seg.scores(gts, preds)

    return val_loss.cpu().numpy() / float(len(data_loader)), score
from core import Total_fusion_loss, Total_fusion_loss2, RGB2YCrCb, Fusionloss, Fusionloss_add, Fusionloss2, Fusionloss3, \
    Fusionloss4, Fusionloss_grad3


def train_seg(cfg,iter_):
    num_workers = 4

    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend=args.backend,)
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = voc2.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu//2,
                              # shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)


    '''
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(torch.cuda.device_count()))
    else:
        print('Using CPU:')
        device = torch.device('cpu')
    '''
    device = torch.device('cuda:0')

    # device  =torch.device(args.local_rank)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    criterion_seg = criterion.to(device)
    model = Network3(cfg.exp.backbone, cfg.dataset.num_classes, 256, True).cuda()
    if  os.path.exists('./checkpoint/model-fusion_add_final2.pth') and iter_>1:
        model.load_state_dict(torch.load('./checkpoint/model-fusion_add_final2.pth'))
    miou = val_segformer(model, 0)
    print('initial_miou',miou)
    param_groups = model.denoise_net.get_param_groups()
    model = model.train()
    model.to(device)
    # optimizer
    it_start = (iter_-1) * 10000
    iter_nums = 10000
    if iter_ == 1:
        # it_start = 0
        iter_nums = 10000

    optimizer = PolyWarmupAdamW_seg(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },

            {
                "params": param_groups[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        iter_curr=it_start,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )


    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    criterion = criterion.to(device)

    # train_sampler.set_epoch(0)
    train_loader_iter = iter(train_loader)

    # for n_iter in tqdm(range(cfg.train.max_iters), total=cfg.train.max_iters, dynamic_ncols=True):
    for n_iter in range(iter_nums):
        try:
            _, inputs_ir, inputs_vis, inputs_mask, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            _, inputs_ir, inputs_vis, inputs_mask, labels = next(train_loader_iter)

        inputs_ir = inputs_ir.to(device, non_blocking=True)
        inputs_vis = inputs_vis.to(device, non_blocking=True)
        inputs_mask = inputs_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs_ir = inputs_ir[:, 0:1, :, :]
        _,__,segmap = model(inputs_mask)
        outputs = F.interpolate(segmap, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_loss =criterion_seg(outputs,labels.type(torch.long))
        optimizer.zero_grad()
        seg_loss.backward()
        optimizer.step()
        if (n_iter + 1) % cfg.train.log_iters == 0 and args.local_rank == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            lr = optimizer.param_groups[0]['lr']
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f" % (
            n_iter + 1, delta, eta, lr, seg_loss.item()))

        if (n_iter + 1) % 1000 == 0:
            miou2 = val_segformer(model, n_iter)
            if miou2>miou:
                torch.save(model.state_dict(),
                           "./checkpoint/model-fusion_add_final2.pth")
                miou = miou2
    miou2 = val_segformer(model,iter_)
    if miou2 > miou:
        torch.save(model.state_dict(),
               "./checkpoint/model-fusion_add_final2.pth")
        miou = miou2
    return True
def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def train_fusion(cfg,iter_ = 1):
    ### training fusion
    num_workers = 4

    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend=args.backend,)
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu//2,
                              #shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    '''
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(torch.cuda.device_count()))
    else:
        print('Using CPU:')
        device = torch.device('cpu')
    '''
    device = torch.device('cuda:0')

    model = Network3(cfg.exp.backbone,cfg.dataset.num_classes).cuda()

    if os.path.exists('./checkpoint/model-fusion_add_final2.pth') and iter_>2:
        model.load_state_dict(torch.load('./checkpoint/model-fusion_add_final2.pth'))
    model2 = Fusion_Network3_ac()
    if os.path.exists('./checkpoint/modelfusion-final2.pth') and iter_>1:
        model2.load_state_dict(torch.load('./checkpoint/modelfusion-final2.pth'))
    model.to(device)
    model2.to(device)
    # if iter_>1:
    #     iter_ = iter_ - 1
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": model2.parameters(),
                "lr": cfg.optimizer.learning_rate/iter_,
                "weight_decay": cfg.optimizer.weight_decay,
            },

        ],
        lr = (3e-4)/iter_,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = (3e-5)/iter_,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )

    criterion_seg = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)

    train_loader_iter = iter(train_loader)

    if iter_ ==1:
        iter_num = 6000
    else:
        iter_num = 4000
    train_loss_buffer = torch.zeros((2, iter_num))
    for n_iter in range(iter_num):
        
        try:
            _, inputs_ir, inputs_vis, inputs_mask, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            _, inputs_ir, inputs_vis, inputs_mask, labels = next(train_loader_iter)

        inputs_ir = inputs_ir.to(device, non_blocking=True)
        inputs_vis = inputs_vis.to(device, non_blocking=True)
        inputs_mask = inputs_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs_ir = inputs_ir[:,0:1,:,:]
        inputs_vis = RGB2YCrCb(inputs_vis)

        with torch.no_grad():
            out0,out1 = model.denoise_net.encoder.forward_fusion(inputs_mask)
        fusion = model2(inputs_ir,inputs_vis,out0,out1)
        optimizer.zero_grad()
        if iter_ > 1:
            fusion_loss = Fusionloss_grad3()
            fused_ycbcr = inputs_vis.clone()
            fused_ycbcr[:, 0:1, :, :] = fusion
            fused_rgb = YCrCb2RGB(fused_ycbcr)
            loss1 = fusion_loss(inputs_ir, inputs_vis, fusion, inputs_mask)
            loss2 = model._loss(fused_rgb, labels, criterion_seg)
            if n_iter > 10:
                train_loss_buffer[0, n_iter] = loss1.item()
                train_loss_buffer[1, n_iter] = loss2.item()
                w_i = torch.Tensor(train_loss_buffer[:, n_iter - 1] / train_loss_buffer[:, n_iter - 2]).cuda()
                batch_weight = 2 * F.softmax(w_i / 1000.0, dim=-1)
                seg_loss = batch_weight[0] * loss1 * (0.4/iter_) + batch_weight[1] * loss2*0.8
                if n_iter > 1 and iter_ > 1 and (n_iter + 1) % 100 == 0:
                    print('batchweight',batch_weight[0], batch_weight[1])
            else:
                train_loss_buffer[0, n_iter] = loss1.item()
                train_loss_buffer[1, n_iter] = loss2.item()
                seg_loss =  (0.4/iter_) * loss1 + 0.8 * loss2
        else:
            fusion_loss = Fusionloss3()
            seg_loss = fusion_loss(inputs_ir, inputs_vis, fusion, inputs_mask)
        seg_loss.backward()
        optimizer.step()
        
        if (n_iter+1) % cfg.train.log_iters == 0 and args.local_rank==0:
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            lr = optimizer.param_groups[0]['lr']
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(n_iter+1, delta, eta, lr, seg_loss.item()))


        if (n_iter + 1) % 50 == 0:
             with torch.no_grad():
                 fusion = model2(inputs_ir, inputs_vis, out0, out1)
                 torchvision.utils.save_image(inputs_ir[:2], 'input_ir_2.png')
                 torchvision.utils.save_image(inputs_vis[:2], 'input_vis_2.png')
                 torchvision.utils.save_image(inputs_mask[:2], 'input_mask_2.png')

                 torchvision.utils.save_image(fusion[:2], 'output_2..png')

        if (n_iter+1)%500 ==0:
            torch.save(model2.state_dict(),
                       "./checkpoint/modelfusion-final2.pth")

    torch.save(model2.state_dict(),
               "./checkpoint/modelfusion-final2.pth")
    print('generate the test samples -------------')
    val_fusion(model2,model, iter_)
    print('generate the train samples -------------')
    val_fusion_train(model2,model,iter_)

    return True


if __name__ == "__main__":
    #
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg_fusion = OmegaConf.load(args.configf)
    if args.local_rank == 0:
        setup_logger()
        logging.info('\nconfigs: %s' % cfg)
    for iter_ in range(1,8):
        if iter_>1:
            print('-------fusion--------')
            train_fusion(cfg_fusion, iter_)
            print('------Segment--------')
            train_seg(cfg, iter_)
        else:
            print('-------fusion--------')
            train_fusion(cfg_fusion, iter_)
            print('------Segment--------')
            train_seg(cfg, iter_)

