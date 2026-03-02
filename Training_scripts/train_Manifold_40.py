import sys
import os

sys.path.append(os.path.abspath("./MeshConv3D/"))

import torch
import torch.nn as nn
import gc
import utils
import mesh_ops
import torch.nn as F
import numpy as np
import mesh_ops
import dataset
from tqdm import tqdm
import time
import trimesh as tm
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime
import os
import mesh_net
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
torch._logging.set_logs(recompiles=True)
# import torch.distributed as dist

def train(rank, world_size, net_params,save_path,training,dataset_path,model_evaluate):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Model initialization
    net = mesh_net.Mesh_Autoencoder(net_params).cuda(rank)
    net = DDP(net, device_ids=[rank])#, output_device=rank)
    
    # Data initialization
    if training:
        train_set = dataset.MeshDataset(dataset_path, K=10, features=['points'],augment=['scale','orient'], nb_augments=10,name='Manifold',common_edge=True,descriptors=False)
    else:
        train_set = dataset.MeshDataset(dataset_path, K=10, features=['points'],train=False,augment=[], nb_augments=1,name='Manifold',common_edge=True,descriptors=False)
        net.load_state_dict(torch.load(model_evaluate))
        net.eval()
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_size=10, num_workers=4, pin_memory=True, collate_fn = utils.collate_fn_dft, sampler=sampler)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-12)
    scaler = torch.cuda.amp.GradScaler()

    epochs=1000
    
    for j in range(epochs):
        losses = []
        losses2,losses3, losses4 = [],[],[]
        cpt=0

        sampler.set_epoch(j)  # Shuffle data for each epoch
        for i in tqdm(train_loader):
            data = i[0]
            data.to(rank)
            label = i[1]

            optimizer.zero_grad(set_to_none=True)
            
            input_vertices = data.vertices.clone().to(rank)
            if not training:
                input_faces = (data.faces.clone().to(rank))

            with torch.cuda.amp.autocast():
                out = net(data)

            target_points, target_faces = out.reconstruct_mesh()

            if training:
                loss = loss_fn(input_vertices,target_points.to(rank))
            else:
                loss = torch.mean(utils.error_meshes_mm(input_vertices,target_points.to(rank)), dim=1)
                for m in range(input_vertices.size(0)):
                    input_mesh = tm.base.Trimesh(input_vertices[m][:-1].clone().cpu().detach().numpy(),input_faces[m][:-1].clone().cpu().detach().numpy())
                    output_mesh = tm.base.Trimesh(target_points[m].clone().cpu().detach().numpy(),target_faces[m][:-1].clone().cpu().detach().numpy())
                    cham, normal = utils.chamfer(torch.Tensor(input_mesh.vertices), torch.Tensor(output_mesh.vertices), input_mesh.vertex_normals, output_mesh.vertex_normals)
                    losses2.append(cham)
                    losses3.append(normal)
                    losses4.append(utils.compute_roughness(output_mesh.vertices, input_mesh.vertices))

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            losses.append(loss.detach())
            
            del loss, out, data, label, i

            cpt+=1

            gc.collect()

        scheduler.step(sum(losses)/cpt)           
        if training:
            print(f"Rank {rank}, Epoch {j}, Loss: {sum(losses)/cpt}")
        else:
            flatten_array = [element.clone().cpu().detach() for sublist in losses for element in sublist]
            flatten_array2 = [element.clone().cpu().detach() for element in losses2]
            flatten_array3 = [element.clone().cpu().detach() for element in losses3]
            flatten_array4 = [element for element in losses4]
            print(f'mm loss {np.mean(np.array(flatten_array))}, rank {rank}')
            print(f'chamfer loss {np.mean(np.array(flatten_array2))}, rank {rank}')
            print(f'normal loss {np.mean(np.array(flatten_array3))}, rank {rank}')
            print(f'CP loss {np.mean(np.array(flatten_array4))}, rank {rank}')
            print()
            break

        if rank == 0 and training:
            torch.save(net.state_dict(),f'{save_path}/MAN40_auto{j}')
    
    # Cleanup
    dist.destroy_process_group()


if __name__== "__main__":
    state=False
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)

    torch.manual_seed(10)
    np.random.seed(10)

    # -------- Modify with your variables here -----------
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12353'
    
    training=True # Whether you want to train or test
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    save_path = f"/home/germainb/PHD_work/MeshConv3D/models/MAN40_auto/MAN40_{date_time}" #Path towards the saved checkpoints of the trained models
    dataset_path = "/home/germainb/PHD_work/Datasets/Manifold_40/Manifold40" #Path towards the source folder of your dataset
    model_evaluate = '' #Model to evaluate in the testing phase
    if training:
        os.mkdir(save_path)

    # Here you can modify the architecture of the model to your liking
    net_params = {'Encoder': {'nb_blocks': 3, 'kernel_size': [10,10,10], 'nb_channels': [64,128,256,9], 'target_size': [440,380,334], 'depth_block': [1,1,1], 'bias': [True,True,True]},
                  'Middle':{'save_latent':False},
                  'Decoder': {'nb_blocks': 3, 'kernel_size': [10,10,10], 'nb_channels': [9,256,128,64], 'target_size': [0,0,0], 'depth_block': [1,1,1], 'bias': [True,True,True]},
                  'Last_Block': {'in_channels':64,'num_params_out':9},
                  'descriptors':False}
    # ------ Modify with your variables here, end --------

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,net_params,save_path,training,dataset_path,model_evaluate,), nprocs=world_size, join=True)