import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch

# ====================================================================
# [全局安全补丁] 彻底解决 PyTorch 2.6 中 weights_only=True 导致的全部报错
# 通过全局拦截 torch.load，一劳永逸地解决 test.py, experimental.py 等所有文件的崩溃问题！
# ====================================================================
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load
# ====================================================================

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.lowlight import End2EndLowLightModel
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)


def remap_checkpoint_state_dict(ckpt_state_dict, model_state_dict, end2end=False):
    remapped_state_dict = {}
    for k, v in ckpt_state_dict.items():
        key = k[7:] if k.startswith('module.') else k
        if key in model_state_dict:
            remapped_state_dict[key] = v
            continue
        if end2end:
            yolo_key = f'yolo_net.{key}'
            if yolo_key in model_state_dict:
                remapped_state_dict[yolo_key] = v
    return remapped_state_dict


def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    plots = not opt.evolve  
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  
    is_coco = opt.data.endswith('coco.yaml')

    loggers = {'wandb': None}  
    if rank in [-1, 0]:
        opt.hyp = hyp  
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  

    nc = 1 if opt.single_cls else int(data_dict['nc'])  
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  

    use_zero_dce = opt.use_zero_dce
    logger.info('Online Zero-DCE is %s', 'enabled' if use_zero_dce else 'disabled')
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  
        ckpt = torch.load(weights)  
        base_yolo = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        model = End2EndLowLightModel(base_yolo, dce_weights=opt.dce_weights).to(device) if use_zero_dce else base_yolo
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  
        state_dict = ckpt['model'].float().state_dict()  
        state_dict = remap_checkpoint_state_dict(state_dict, model.state_dict(), end2end=use_zero_dce)
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  
        model.load_state_dict(state_dict, strict=False)  
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  
    else:
        base_yolo = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        model = End2EndLowLightModel(base_yolo, dce_weights=opt.dce_weights).to(device) if use_zero_dce else base_yolo
    
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  
    train_path = data_dict['train']
    test_path = data_dict['val']

    if use_zero_dce:
        freeze = [f'yolo_net.model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    else:
        freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True  
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    nbs = 64  
    accumulate = max(round(nbs / total_batch_size), 1)  
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  
            
        # 注意力机制权重衰减配置
        for attr in ['im', 'imc', 'imb', 'imo', 'ia']:
            if hasattr(v, attr):
                obj = getattr(v, attr)
                if hasattr(obj, 'implicit'):
                    pg0.append(obj.implicit)
                else:
                    for iv in obj:
                        pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'): pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'): pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'): pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'): pg0.append(v.attn.relative_position_bias_table)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  
    optimizer.add_param_group({'params': pg2})  
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model) if rank in [-1, 0] else None

    start_epoch, best_fitness = 0, 0.0
    if pretrained and opt.resume:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  

        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  

        del ckpt, state_dict
    elif pretrained:
        del ckpt, state_dict

    gs = max(int(model.stride.max()), 32)  
    nl = model.model[-1].nl  
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  

    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  
    nb = len(dataloader)  
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  
            if plots:
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  

    # DDP 初始化
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)

    hyp['box'] *= 3. / nl  
    hyp['cls'] *= nc / 80. * 3. / nl  
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  
    model.hyp = hyp  
    model.gr = 1.0  
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  
    model.names = names

    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  
    maps = np.zeros(nc)  
    results = (0, 0, 0, 0, 0, 0, 0)  
    scheduler.last_epoch = start_epoch - 1  
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)  
    compute_loss = ComputeLoss(model)  
    
    # ------------------ 早停机制参数 ------------------
    patience = 30   # 连续 30 个 epoch 指标没提升就停止训练
    best_fitness_epoch = 0  
    stop_training = False  
    # ------------------------------------------------
    
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    torch.save(model, wdir / 'init.pt')
    
    for epoch in range(start_epoch, epochs):  # epoch 大循环 ===================================
        model.train()

        if opt.image_weights:
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=device)  
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch 循环 ------------------------------------
            ni = i + nb * epoch  
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  

            if ni <= nw:
                xi = [0, nw]  
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  
                sf = sz / max(imgs.shape[2:])  
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            with amp.autocast(enabled=cuda):
                pred = model(imgs)  
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  
                if rank != -1:
                    loss *= opt.world_size  
                if opt.quad:
                    loss *= 4.

            scaler.scale(loss).backward()

            if ni % accumulate == 0:
                scaler.step(optimizer)  
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

        # end batch ------------------------------------------------------------------------------------

        lr = [x['lr'] for x in optimizer.param_groups]  
        scheduler.step()

        if rank in [-1, 0]:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric)

            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  

            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  
                    'x/lr0', 'x/lr1', 'x/lr2']  
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  

            # ------------------ 早停与最佳模型保存逻辑 ------------------
            # [毕设关键]：仅按 mAP@0.5 (results[2]) 保存 best.pt！
            fi = results[2]  
            if fi > best_fitness:
                best_fitness = fi
                best_fitness_epoch = epoch
            wandb_logger.end_epoch(best_result=best_fitness == fi)
            
            if (epoch - best_fitness_epoch) >= patience:
                logger.info(f'\n[Early Stopping] 触发早停！连续 {patience} 轮无提升，最佳 epoch 为 {best_fitness_epoch}。')
                stop_training = True
            # ---------------------------------------------------------

            if (not opt.nosave) or (final_epoch and not opt.evolve):  
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # 将主进程的停止信号广播给所有 GPU，防止进程死锁崩溃！
        stop_tensor = torch.tensor([1 if stop_training else 0], dtype=torch.int32, device=device)
        if opt.world_size > 1:
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.SUM)
        if stop_tensor.item() > 0:
            break  # 所有 GPU 同步跳出 epoch 大循环
        
    # end epoch 大循环 =========================================================================

    if rank in [-1, 0]:
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        
        # ------------------ 1. 先强制生成最佳权重的所有评估图表 ------------------
        if best.exists():
            logger.info(f'\n[Final Evaluation] 正在测试最佳权重 {best} 并生成 P/R/F1/PR 曲线等完整图表...')
            test.test(data_dict,
                      batch_size=batch_size * 2,
                      imgsz=imgsz_test,
                      conf_thres=0.001,
                      iou_thres=0.65,
                      model=attempt_load(best, device).half(),
                      single_cls=opt.single_cls,
                      dataloader=testloader,
                      save_dir=save_dir,
                      save_json=False,
                      plots=True,  # 强制生成所有需要的曲线图！
                      is_coco=is_coco,
                      v5_metric=opt.v5_metric)
        # ----------------------------------------------------------------------
        
        # ------------------ 2. 再将刚刚生成的图表上传到 W&B ------------------
        if plots:
            plot_results(save_dir=save_dir)  
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # ----------------------------------------------------------------------
        
        final = best if best.exists() else last  
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  
        if wandb_logger.wandb and not opt.evolve:  
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--use-zero-dce', action='store_true', help='enable online Zero-DCE + YOLOv7 end-to-end training')
    parser.add_argument('--dce-weights', type=str, default='Epoch99.pth', help='pretrained Zero-DCE weights path')
    opt = parser.parse_args()

    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    opt.local_rank = int(os.environ.get('LOCAL_RANK', opt.local_rank))
    set_logging(opt.global_rank)

    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  

    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    
    if 'LOCAL_RANK' in os.environ:
        opt.local_rank = int(os.environ['LOCAL_RANK'])
        
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  

    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  
        train(hyp, opt, device, tb_writer)
