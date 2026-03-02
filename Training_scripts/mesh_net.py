from typing import List
import time
import torch
import torch.nn as nn
import utils
import mesh_ops
import dataset
import trimesh as tm
import time
import subprocess
import os
import prepare_dataset
import trimesh
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

    
class Encoder_block(nn.Module):
    def __init__(self, kernel_size, in_channels, middle_channels, out_channels, target_size, depth_block, bias=True, store_connectivity=False):
        super().__init__()
        
        self.convs = nn.Sequential()
        self.convs.append(mesh_ops.MeshConv_2(kernel_size,in_channels,middle_channels, bias))
        self.convs.append(mesh_ops.MeshBatchNorm(middle_channels))
        self.convs.append(mesh_ops.ReLU())
        for i in range(depth_block-1):
            self.convs.append(mesh_ops.MeshConv_2(kernel_size,middle_channels,middle_channels, bias))
            self.convs.append(mesh_ops.MeshBatchNorm(middle_channels))
            self.convs.append(mesh_ops.ReLU())

        self.pool = mesh_ops.MeshPool_Parrallel(kernel_size,target_size=target_size,store_connectivity=store_connectivity)


    def forward(self, mesh):

        mesh = self.convs(mesh)
        mesh = self.pool(mesh)

        return mesh

class Encoder(nn.Module):
    """
    Encoder:
        nb_blocks
        kernel_size
        nb_channels size=nb_blocks + 1
        depth block
        bias
    """

    def __init__(self, params):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(params['nb_blocks']):
            self.blocks.append(Encoder_block(params['kernel_size'][i],params['nb_channels'][(i)], params['nb_channels'][(i)+1],params['nb_channels'][(i)+1], params['target_size'][i],params['depth_block'][i], params['bias'][i]))
    
    def forward(self, mesh):
        for i in self.blocks:
            mesh = i(mesh)

        return mesh
    
class Middle_Net_mesh(nn.Module):
    def __init__(self, params_middle):
        super().__init__()
        self.save = params_middle['save_latent']
    
    def forward(self, mesh):
        # reconstruction module
        vertices,faces = mesh.reconstruct_mesh()

        if self.save:
            mesh.latent_verts = vertices
            mesh.latent_faces = faces
            mesh.latent_adj = mesh.adjacency.clone().type(torch.int)

        mesh[1] = vertices[torch.arange(faces.size(0))[:,None,None],faces].reshape(faces.size(0),faces.size(1),9)

        return mesh

class Decoder_block(nn.Module):
    def __init__(self, kernel_size, in_channels, middle_channels, out_channels, target_size, depth_block, bias=True):
        super().__init__()
        
        self.convs = nn.Sequential()
        self.convs.append(mesh_ops.MeshConv_2(kernel_size,in_channels,middle_channels, bias))
        self.convs.append(mesh_ops.MeshBatchNorm(middle_channels))
        self.convs.append(mesh_ops.ReLU())
        for i in range(depth_block-1):
            self.convs.append(mesh_ops.MeshConv_2(kernel_size,middle_channels,middle_channels, bias))
            self.convs.append(mesh_ops.MeshBatchNorm(middle_channels))
            self.convs.append(mesh_ops.ReLU())

        self.unpool = mesh_ops.MeshUnPool_memory(kernel_size)

    def forward(self, mesh):

        mesh = self.unpool(mesh)
        mesh = self.convs(mesh)

        return mesh
    
class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(params['nb_blocks']):
            self.blocks.append(Decoder_block(params['kernel_size'][i],params['nb_channels'][(i)], params['nb_channels'][(i)+1],params['nb_channels'][(i)+1], params['target_size'][i],params['depth_block'][i], params['bias'][i]))
    
    def forward(self, mesh):
        for i in self.blocks:
            mesh = i(mesh)

        return mesh

class Linear_Block(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.blocks = nn.Sequential()

        self.blocks.append(mesh_ops.MeshConv_2(1, params['in_channels'], params['in_channels']))
        self.blocks.append(mesh_ops.MeshBatchNorm(params['in_channels']))
        self.blocks.append(mesh_ops.ReLU())
        
        self.blocks.append(mesh_ops.MeshConv_2(1, params['in_channels'], params['num_params_out']))
    
    def forward(self, mesh):
        for i in self.blocks:
            mesh = i(mesh)
        
        return mesh

class Mesh_Autoencoder(nn.Module):
    def __init__(self, params):
        super(Mesh_Autoencoder, self).__init__()

        self.fc = mesh_ops.MeshLinear(9, params['Encoder']['nb_channels'][0])
        self.relu = mesh_ops.ReLU()
        
        self.encoder = Encoder(params['Encoder'])

        self.middle = Middle_Net_mesh(params['Middle'])

        self.decoder = Decoder(params['Decoder'])
        self.last_block_dec = Linear_Block(params['Last_Block'])
    

    def forward(self, mesh):
        self.device = mesh[0].device

        mesh = self.fc(mesh)
        mesh = self.relu(mesh)

        mesh.compute_ring_k_batch(10)

        mesh = self.encoder(mesh)
        # Bottleneck
        mesh = self.middle(mesh)
        # End Bottlencek
        mesh = self.decoder(mesh)
        mesh = self.last_block_dec(mesh)

        return mesh