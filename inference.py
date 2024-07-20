from predict import Predictor
from config.dataconfig import Config as DataConfig
from data import create_dataloader
import argparse
import torch
from tqdm import tqdm
from diffusion import ddim_steps
from PIL import Image
import numpy as np
import os
from train import init_distributed

def make_dir(sav_root: str):
    if not os.path.exists(sav_root):
        os.makedirs(sav_root)

if __name__ == "__main__":

    init_distributed()

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--DataConfigPath', type=str, default='./config/data.yaml')
    parser.add_argument('--dataset_path', type=str, default='./dataset/deepfashion')
    parser.add_argument('--val_savedir', type=str, default='./infer_results')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--expr_name', type=str, required=True)
    parser.add_argument('--use_warp', action= 'store_true')
    parser.add_argument('--valimg_num', type=int, default= 8570)
    parser.add_argument('--val_batch_size', type=int, default= 10)

    args = parser.parse_args()
    DataConf = DataConfig(args.DataConfigPath)
    DataConf.data.path = args.dataset_path
    # DataConf.data.val.batch_size = 1 # for the validation dataset
    DataConf.data.val.batch_size = args.val_batch_size
    assert args.valimg_num % args.val_batch_size == 0, 'The val images generated must be divisable by val batch size'
    make_dir(os.path.join(args.val_savedir, args.expr_name))
    

    obj = Predictor(ckpt_path= f"./checkpoints/{args.expr_name}/last.pt", args= args)
    val_dataset = create_dataloader(DataConf.data, distributed= True, labels_required = True, is_inference=True,) # test dataset


    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataset)):

            if i >= (args.valimg_num // args.val_batch_size):
                break

            val_img = data['source_image'].cuda()
            val_pose = data['target_skeleton'].cuda()
            if args.use_warp:
                # target_pose = torch.cat([batch['target_skeleton'], batch['source_skeleton']], 0)
                val_pose = torch.cat([val_pose, data['source_skeleton'].cuda()], 0)
            fname = data['path']

            fname_states = [] # True means the img is in the sav_list
            for idx in range(len(fname)):
                if fname[idx] in os.listdir(os.path.join(args.val_savedir, args.expr_name)):
                    fname_states.append(True)
                else:
                    fname_states.append(False)
            
            if all(fname_states):
                continue
            elif True in fname_states:
                indexes = [idx for idx, item in enumerate(fname_states) if item == True]
                assert all(fname_states[:indexes[-1]+1]) == True
                val_img = val_img[indexes[-1]+1:, ...]
                val_pose = torch.cat([data['target_skeleton'].cuda()[indexes[-1]+1:, ...], 
                                      data['source_skeleton'].cuda()[indexes[-1]+1:, ...]], 0)
                fname = fname[indexes[-1]+1:]

            # print ('Sampling algorithm used: DDIM')
            nsteps = 50
            noise = torch.randn(val_img.shape).cuda() # torch.Size([4, 3, 256, 256])
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds, flow_fields, feature_map_out = ddim_steps(noise, seq, obj.model, obj.betas.cuda(), [val_img, val_pose])
            samples = xs[-1]

            # import pdb; pdb.set_trace()
            samples = (torch.clamp(samples, -1., 1.) + 1.0) / 2.0
            numpy_imgs = samples.permute(0,2,3,1).detach().cpu().numpy()
            fake_imgs = (255*numpy_imgs).astype(np.uint8)
            sav_path = os.path.join(args.val_savedir, args.expr_name)
            # Image.fromarray(fake_imgs[0]).save(os.path.join(sav_path, fname[0]))
            [Image.fromarray(im).save(os.path.join(sav_path, fname[idx])) for idx, im in enumerate(fake_imgs)]

#  obj.predict_pose(image=<PATH_OF_SOURCE_IMAGE>, sample_algorithm='ddim', num_poses=4, nsteps=50)