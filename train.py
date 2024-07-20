import os
import warnings

warnings.filterwarnings("ignore")

import time, cv2, torch, wandb
import torch.distributed as dist
from config.diffconfig import DiffusionConfig, get_model_conf
from config.dataconfig import Config as DataConfig
from tensorfn import load_config as DiffConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
from tensorfn.optim import lr_scheduler
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import data as deepfashion_data
from model import UNet
from utils import util

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # import pdb; pdb.set_trace()

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_main_process():
    try:
        if dist.get_rank()==0:
            return True
        else:
            return False
    except:
        return True

def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)




def train(conf, loader, val_loader, model, ema, diffusion, betas, optimizer, scheduler, guidance_prob, cond_scale, device, wandb, **kwargs):

    import time

    i = 0 if conf.ckpt is None else kwargs.get('start_epoch') * len(loader)

    loss_list = []
    loss_mean_list = []
    loss_vb_list = []
    loss_correct_list = []
    loss_regular_list = []
    flow2color = util.flow2color() # create the class for flow2color

    start_epoch = kwargs.get('start_epoch') if conf.ckpt is not None else 0 
    end_epoch = kwargs.get('end_epoch') if conf.ckpt is not None else 30
    print(f"The end epoch is {end_epoch-1}")
 
    for epoch in range(start_epoch, end_epoch):

        if is_main_process: print ('#Epoch - '+str(epoch))

        start_time = time.time()

        for batch in tqdm(loader):

            i = i + 1

            img = torch.cat([batch['source_image'], batch['target_image']], 0)
            target_img = torch.cat([batch['target_image'], batch['source_image']], 0)
            target_pose = torch.cat([batch['target_skeleton'], batch['source_skeleton']], 0)

            img = img.to(device)
            target_img = target_img.to(device)
            target_pose = target_pose.to(device)
            time_t = torch.randint(
                0,
                conf.diffusion.beta_schedule["n_timestep"], # 1000
                (img.shape[0],),
                device=device,
            )

            loss_dict = diffusion.training_losses(model, x_start = target_img, t = time_t, cond_input = [img, target_pose], prob = 1 - guidance_prob)
            # for each item in loss_dict, shape is torch.Size([4])

            loss = loss_dict['loss'].mean()
            loss_mse = loss_dict['mse'].mean()
            loss_vb = loss_dict['vb'].mean()
            if ema.conf.use_warp:
                loss_correct = loss_dict['correctness_gen'] # mean
                loss_regular = loss_dict['regularization'] # mean
                loss = loss + loss_correct + loss_regular
        

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1) # clips the gradients of a set of parameters to a specified maximum norm(1 in this code)
            scheduler.step()
            optimizer.step()
            loss = loss_dict['loss'].mean()

            loss_list.append(loss.detach().item())
            loss_mean_list.append(loss_mse.detach().item())
            loss_vb_list.append(loss_vb.detach().item())
            if ema.conf.use_warp:
                loss_correct_list.append(loss_correct.detach().item())
                loss_regular_list.append(loss_regular.detach().item())


            accumulate(
                ema, model.module, 0 if i < conf.training.scheduler.warmup else 0.9999
            )


            if i%args.save_wandb_logs_every_iters == 0 and is_main_process():

                if ema.conf.use_warp:

                    wandb.log({'loss':(sum(loss_list)/len(loss_list)), 
                               'loss_vb':(sum(loss_vb_list)/len(loss_vb_list)), 
                               'loss_mean':(sum(loss_mean_list)/len(loss_mean_list)),
                               'loss_correct':(sum(loss_correct_list)/len(loss_correct_list)),
                               'loss_regular':(sum(loss_regular_list)/len(loss_regular_list)),
                                'epoch':epoch,'steps':i})
                else:
                    wandb.log({'loss':(sum(loss_list)/len(loss_list)), 
                               'loss_vb':(sum(loss_vb_list)/len(loss_vb_list)), 
                               'loss_mean':(sum(loss_mean_list)/len(loss_mean_list)),
                                'epoch':epoch,'steps':i})
                loss_list = []
                loss_mean_list = []
                loss_vb_list = []
                loss_correct_list = []
                loss_regular_list = []


            if i%args.save_checkpoints_every_iters == 0 and is_main_process():

                # if conf.distributed:
                #     model_module = model.module

                # else:
                #     model_module = model

                # torch.save(
                #     {
                #         "model": model_module.state_dict(),
                #         "ema": ema.state_dict(),
                #         "scheduler": scheduler.state_dict(),
                #         "optimizer": optimizer.state_dict(),
                #         "conf": conf,
                #     },
                #     conf.training.ckpt_path + f"/model_{str(i).zfill(6)}.pt"
                # )
                pass

        if is_main_process():

            print ('Epoch Time '+str(int(time.time()-start_time))+' secs')
            print ('Model Saved Successfully for #epoch '+str(epoch)+' #steps '+str(i))

            if conf.distributed:
                model_module = model.module

            else:
                model_module = model

            torch.save(
                {
                    "model": model_module.state_dict(),
                    "ema": ema.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "conf": conf,
                },
                conf.training.ckpt_path + '/last.pt'
               
            )
            torch.save(
                {
                    "model": model_module.state_dict(),
                    "ema": ema.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "conf": conf,
                },
                conf.training.ckpt_path + f'/Epoch{epoch}.pt'
               
            )

        if (epoch)%args.save_wandb_images_every_epochs==0:

            print ('Generating samples at epoch number ' + str(epoch))

            val_batch = next(val_loader) # only generate 1 batch of image
            val_img = val_batch['source_image'].cuda()
            val_pose = val_batch['target_skeleton'].cuda()
            val_pose_source = val_batch['source_skeleton'].cuda()
            val_pose = torch.cat([val_pose, val_pose_source], dim = 0)

            with torch.no_grad():

                if args.sample_algorithm == 'ddpm':
                    print ('Sampling algorithm used: DDPM')
                    samples = diffusion.p_sample_loop(ema, x_cond = [val_img, val_pose], progress = True, cond_scale = cond_scale)
                elif args.sample_algorithm == 'ddim':
                    print ('Sampling algorithm used: DDIM')
                    nsteps = 50
                    noise = torch.randn(val_img.shape).cuda() # torch.Size([4, 3, 256, 256])
                    seq = range(0, 1000, 1000//nsteps)
                    xs, x0_preds, flow_fields, feature_map_out = ddim_steps(noise, seq, ema, betas.cuda(), [val_img, val_pose])
                    samples = xs[-1].cuda()


            grid = torch.cat([val_img, val_pose[:val_pose.shape[0]//2,:3], samples], -1)
            
            gathered_samples = [torch.zeros_like(grid) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, grid) 

            if flow_fields is not None:
                
                if len(flow_fields) == 3:
                    flow_maps1, flow_maps2, flow_maps3 = flow_fields
                elif len(flow_fields) == 2:
                    flow_maps1, flow_maps2 = flow_fields
                    flow_maps3 = None
                elif len(flow_fields) == 1:
                    flow_maps1 = flow_fields[0]
                    flow_maps2 = None
                    flow_maps3 = None

                grid_map1 = torch.cat([flow2color(flow_map1).unsqueeze(0) for flow_map1 in flow_maps1], 0)

                if flow_maps2 is not None:
                    grid_map2 = torch.cat([flow2color(flow_map2).unsqueeze(0) for flow_map2 in flow_maps2], 0)
                    gathered_samples_flow2 = [torch.zeros_like(grid_map2) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_samples_flow2, grid_map2)
                else:
                    grid_map2 = None
                
                if flow_maps3 is not None:
                    grid_map3 = torch.cat([flow2color(flow_map3).unsqueeze(0) for flow_map3 in flow_maps3], 0)
                    gathered_samples_flow3 = [torch.zeros_like(grid_map3) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_samples_flow3, grid_map3)
                else:
                    grid_map3 = None
                
                gathered_samples_flow1 = [torch.zeros_like(grid_map1) for _ in range(dist.get_world_size())]
                
                dist.all_gather(gathered_samples_flow1, grid_map1)
                

            if feature_map_out is not None:
                grid_feature_map = torch.cat(feature_map_out, -1)
                gathered_samples_map = [torch.zeros_like(grid_feature_map) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples_map, grid_feature_map)
            

            if is_main_process():
                
                wandb.log({'samples':wandb.Image(torch.cat(gathered_samples, -2))})
                if flow_fields is not None:
                    wandb.log({'flowmap1':wandb.Image(torch.cat(gathered_samples_flow1, -2))})
                    if flow_maps2 is not None:
                        wandb.log({'flowmap2':wandb.Image(torch.cat(gathered_samples_flow2, -2))})
                    if flow_maps3 is not None:
                        wandb.log({'flowmap3':wandb.Image(torch.cat(gathered_samples_flow3, -2))})

                if feature_map_out is not None:
                    img = (torch.cat(gathered_samples_map, -2)).unsqueeze(1)
                    wandb.log({'feature_map':wandb.Image(img)})

            del val_img
            del val_pose_source
            del val_pose
            del noise
            del xs
            del grid
            del gathered_samples
            del flow_fields
            del flow_maps1
            del flow_maps2
            del grid_map1
            del gathered_samples_flow1
            del grid_map2
            del gathered_samples_flow2
            if flow_maps3 is not None:
                del flow_maps3
                del grid_map3
                del gathered_samples_flow3
            torch.cuda.empty_cache()


def main(settings, EXP_NAME):

    [args, DiffConf, DataConf] = settings

    if is_main_process(): wandb.init(project="person-synthesis", name = EXP_NAME,  settings = wandb.Settings(code_dir="."))

    if DiffConf.ckpt is not None: 
        DiffConf.training.scheduler.warmup = 0

    DiffConf.distributed = True
    local_rank = int(os.environ['LOCAL_RANK'])
    
    DataConf.data.train.batch_size = args.batch_size//2  #src -> tgt , tgt -> src
    
    val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = True)
    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    val_dataset = iter(cycle(val_dataset))

    model = get_model_conf().make_model()
    model = model.to(args.device)
    ema = get_model_conf().make_model()
    ema = ema.to(args.device)

    if DiffConf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True
        )

    optimizer = DiffConf.training.optimizer.make(model.parameters())
    scheduler = DiffConf.training.scheduler.make(optimizer)

    if DiffConf.ckpt is not None:
        ckpt = torch.load(DiffConf.ckpt, map_location=lambda storage, loc: storage)

        if DiffConf.distributed:
            model.module.load_state_dict(ckpt["model"])

        else:
            model.load_state_dict(ckpt["model"])

        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

        if is_main_process():  print ('model loaded successfully')

    betas = DiffConf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, get_model_conf().flownet_attnlayer, predict_xstart = False)

    train(
        DiffConf, train_dataset, val_dataset, model, ema, diffusion, betas, optimizer, scheduler, args.guidance_prob, args.cond_scale, args.device, wandb, start_epoch = args.start_epoches, end_epoch = args.end_epoches
    )

if __name__ == "__main__":

    init_distributed()

    import argparse

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='pidm_deepfashion')
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf')
    parser.add_argument('--DataConfigPath', type=str, default='./config/data.yaml')
    parser.add_argument('--dataset_path', type=str, default='./dataset/deepfashion')
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--cond_scale', type=int, default=2)
    parser.add_argument('--guidance_prob', type=int, default=0.1)
    parser.add_argument('--sample_algorithm', type=str, default='ddim') # ddpm, ddim
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--save_wandb_logs_every_iters', type=int, default=50)
    parser.add_argument('--save_checkpoints_every_iters', type=int, default=12000)
    parser.add_argument('--save_wandb_images_every_epochs', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_gpu', type=int, default=8)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--resume_train', action= 'store_true')
    parser.add_argument('--start_epoches', type=int, default= 0)
    parser.add_argument('--end_epoches', type=int, default= 30)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print ('Experiment: '+ args.exp_name)

    DiffConf = DiffConfig(DiffusionConfig,  args.DiffConfigPath, args.opts, False)
    DataConf = DataConfig(args.DataConfigPath)

    ckpt_sav_path = util.check_path(args.save_path, args.exp_name) if not args.resume_train else os.path.join(args.save_path, args.exp_name)
    DiffConf.training.ckpt_path = ckpt_sav_path
    DataConf.data.path = args.dataset_path

    if is_main_process():

        if not os.path.isdir(args.save_path): os.mkdir(args.save_path)
        if not os.path.isdir(DiffConf.training.ckpt_path): os.mkdir(DiffConf.training.ckpt_path)

    if args.resume_train:
        DiffConf.ckpt = os.path.join(ckpt_sav_path, 'last.pt')

    main(settings = [args, DiffConf, DataConf], EXP_NAME = args.exp_name)
