from typing import Optional
import torch
import torch.nn as nn
import utils
import math
import mesh_pool
import time
import mesh_unpool
import numpy as np
import torch.nn.functional as F


class MeshConv_2(nn.Module):
    """
    Convolution on a target face and its K neighboring faces. Inspired by subdivnet operation.
    """

    def __init__(self, K, in_channels, out_channels, bias=True, dilation=1, stride=0, ones=False):
        """
        K: kernel size, K>4
        in_channels: number of channels in the input of the convolution
        out_channels: number of channels in th output of the convolution
        padding: int, size of the padding for the convolution
        stride: int, size of the stride for the convolution
        """
        super(MeshConv_2, self).__init__()

        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.stride = stride

        if self.K>1:
            self.conv = nn.Conv2d(in_channels,out_channels,(1,4), bias=bias)#self.in_channels)
        else:
            self.conv = nn.Conv2d(in_channels,out_channels, 1, bias=bias)#self.in_channels)
    
    def forward(self, data):
        """
        Compute the convolution on the faces of the meshes in the batch. 
        The Kernel corresponds to the K-ring neighborhood of the target face

        features :  face features of the meshes (num_meshes, num_faces, in_channels)
        ring_k :    faces in a k_ring neighborhood of the target for 
                    all the faces of all the meshes (num_meshes, num_faces, K)
        """

        features = data[1]
        ring_k = data[3][:,:,:self.K*self.dilation]
        num_meshes, num_faces, num_channels = features.size()

        # Gather features at face neighbors
        neighbor_fea = []
        for i in range(self.K):
            neighbor_fea.append(features[torch.arange(num_meshes)[:,None],ring_k[:,:,i*self.dilation]].transpose(1,2))
        
        neighbor_fea = torch.stack(neighbor_fea,3)

        features_conv = []
        #e_i
        features_conv.append(features[torch.arange(num_meshes)[:,None],ring_k[:,:,0]].transpose(1,2))
        if self.K>1:
            #e_j
            features_conv.append(neighbor_fea[:,:,:,1:].sum(dim=3))
            #|e_i - e_j|
            features_conv.append((neighbor_fea[:,:,:,0].unsqueeze(3) - neighbor_fea[:,:,:,1:]).sum(dim=3).abs())
            #|e_k - e_j| ; k = 1,...,n ; j = 1,...,n            
            temp_feat = [(neighbor_fea[:,:,:,i].unsqueeze(3) - neighbor_fea[:,:,:,i+1:]).abs().sum(dim=3) for i in range(1,self.K-1)]
            features_conv.append(torch.stack(temp_feat,3).sum(3))

        features_conv = torch.stack(features_conv,3)

        data[1] = self.conv(features_conv)[:,:,:,0].transpose(1,2)
        data[1][data[-1][:,0], data[-1][:,1]] = 0

        del neighbor_fea, features, ring_k, features_conv
        torch.cuda.empty_cache()

        return data 

class MeshBatchNorm(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MeshBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x[1] = x[1].transpose(1,2)
        erased = x[-1]
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
                    
        # calculate running estimates
        if self.training:

            num_meshes, num_channels, num_faces = x[1].size()
            mask = torch.ones(x[1].numel(), dtype=torch.bool).view(num_meshes,num_channels,num_faces)

            mask[erased[:,0],:,erased[:,1]] = False
            x_erased = x[1].transpose(1,2)[mask.transpose(1,2)].view(-1,self.num_features)

            mean = x_erased.mean(0)
            n = x_erased.shape[0]

            x_erased = x[1].clone().to(x[1].device)
            x_erased[erased[:,0],:,erased[:,1]] = mean[None,:]

            temp = (x_erased - mean[None,:,None])**2
            var = temp.sum([0,2]) / n

            with torch.no_grad():
                self.running_mean.copy_(exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean)
                # update running_var with unbiased var
                self.running_var.copy_(exponential_average_factor * var \
                    + (1 - exponential_average_factor) * self.running_var)

        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x[1] - mean.view(1,-1,1)) / (torch.sqrt(var.view(1,-1,1) + self.eps))
        if self.affine:
            x[1] = x_hat * self.weight[None, :, None] + self.bias[None, :, None]

        x[1][erased[:,0], :, erased[:,1]] = 0
        x[1] = x[1].transpose(1,2)
        return x

class MeshPool_Parrallel(nn.Module):
    """
    Face pooling of a mesh by face colapse.
    K           : size of the convolution kernel that we need to find at the end of the pooling
    points      : Is the operation applied on the points themselves or on features of the faces
    target_size : The target size of the meshes at the end of the operation
    """

    def __init__(self, K, points = False, target_size=50, store_connectivity=False):
        super(MeshPool_Parrallel, self).__init__()
        self.K = K
        self.choice_points = points
        self.device = None
        self.target_size=target_size
        self.store_connectivity = store_connectivity

    def forward(self, data):
        """
        Select faces to pool according to the features of this face, and update the whole mesh after the face pooling
        The whole pooling is based on the three regions of the chosen 'center' face (the face that will be collapsed).
        The three neighbors of this face are called regions, and we need to find the two faces for each region that will be
        the new neighbors in the end of the operation.
        """

        data[1], data[3], data[0] = data[1].clone().to(self.device), data[3].clone().to(self.device), data[0].clone().to(self.device)
        self.device = data[0].device

        erased = data[-1]
        faces = data[0]
        features = data[1]#.clone().to(self.device).detach()
        adjacency_matrix = data[2]
        ring_k = data[3]
        num_meshes, num_channels = features.shape[0], features.shape[2]
        #Tensors used to restore the connectivity during the unpooling operation
        old_faces = faces.clone().type(torch.int).to(self.device)
        erased_pool = torch.Tensor().to(self.device).type(torch.int)
        adj_face_1_pool = torch.Tensor().to(self.device).type(torch.int)
        adj_face_2_pool = torch.Tensor().to(self.device).type(torch.int)
        reg1_pool = torch.Tensor().to(self.device).type(torch.int)
        reg2_pool = torch.Tensor().to(self.device).type(torch.int)
        indices_iter = torch.Tensor().to(self.device).type(torch.int)
        indices_iter1 = torch.Tensor().to(self.device).type(torch.int)
        indices_iter2 = torch.Tensor().to(self.device).type(torch.int)
        indices_pool = torch.Tensor().to(self.device).type(torch.int)
        inverse_indices1 = torch.Tensor().to(self.device).type(torch.int)
        inverse_indices2 = torch.Tensor().to(self.device).type(torch.int)
        erased = erased.unique(dim=0)

        # Select disjoint pooling regions according to face weights
        if self.choice_points:
            idx, sorted = mesh_pool.compute_weights_points(features, ring_k, num_meshes, features.shape[1], num_channels,erased)

        else:
            idx,sorted = mesh_pool.compute_weights(features, ring_k, num_meshes, features.shape[1], num_channels,erased)
        
        # If mesh.faces - errased_faces of the mesh > target_size -> no need to pool the mesh
        unique,counts = erased[:,0].unique(return_counts = True)
        meshes_to_pool = torch.arange(features.shape[0]).to(self.device)
        num_faces = (features.shape[1] - counts)[meshes_to_pool].type(torch.int)
        meshes_not_poolable = torch.Tensor().to(self.device)
        
        cpt = 0
        while meshes_to_pool.shape[0]>0:
            erased_iter = torch.cat((meshes_to_pool.unsqueeze(1), (torch.ones(meshes_to_pool.shape[0],1) * -1).to(self.device)),dim=1).type(torch.int)
            
            indices = torch.cat([torch.ones(num_faces[i]).to(self.device) * meshes_to_pool[i] for i in range(meshes_to_pool.shape[0])]).type(torch.int)
            faces_corresponding = torch.cat([torch.arange(num_faces[i]).to(self.device) for i in range(meshes_to_pool.shape[0])]).type(torch.int)
            head_meshes = torch.cat((torch.zeros(1).to(self.device), num_faces[:-1].cumsum(dim=0))).type(torch.int)

            # Create regions for all the faces of the meshes to pool
            first_part = torch.cat((torch.arange(adjacency_matrix.size(1)).to(self.device)[idx[indices, faces_corresponding]].unsqueeze(1),adjacency_matrix[indices, idx[indices, faces_corresponding]]), dim=1)

            regions1 = torch.cat((first_part, adjacency_matrix[indices, idx[indices, faces_corresponding], 0].unsqueeze(1)), dim = 1)
            regions1 = torch.cat((regions1.unsqueeze(0), regions1.unsqueeze(0)), dim = 0)
            regions2 = torch.cat((first_part, adjacency_matrix[indices, idx[indices, faces_corresponding], 1].unsqueeze(1)), dim = 1)
            regions2 = torch.cat((regions2.unsqueeze(0), regions2.unsqueeze(0)), dim = 0)
            regions3 = torch.cat((first_part, adjacency_matrix[indices, idx[indices, faces_corresponding], 2].unsqueeze(1)), dim = 1)
            regions3 = torch.cat((regions3.unsqueeze(0), regions3.unsqueeze(0)), dim = 0)

            regions = torch.cat((regions1.unsqueeze(0), regions2.unsqueeze(0), regions3.unsqueeze(0)), dim=0).type(torch.long).to(self.device)
            
            # faces of particular meshes where we don't continue to search for adjacent faces (they are a dead end and we would be looking in circles if continued)
            faces_non_iterate = torch.Tensor(0,2).type(torch.int).to(self.device)

            # Here we look for the adj faces while updating ids and faces_non_iterate
            adj_face_1, adj_face_2, regions, ids, faces_non_iterate = mesh_pool.find_faces_staying(adjacency_matrix, indices, regions, indices.shape[0], 1, faces_non_iterate, self.device)
            
            adj_face_1, adj_face_2, regions, ids, faces_non_iterate = mesh_pool.find_adj_faces_adjacent_region2(adjacency_matrix, indices, adj_face_1, adj_face_2, regions, ids, faces_non_iterate, self.device)
            
            faces_non_iterate = mesh_pool.check_zone_is_not_manifold(adjacency_matrix,indices,adj_face_1,adj_face_2,regions,ids,torch.Tensor(0,2).type(torch.int).to(self.device),self.device)
            # Now we have the adj_faces for every face of the mesh, we need to select the pooling regions so that they are never overlapping
            ret_glob_regions = mesh_pool.create_global_regions(adj_face_1, adj_face_2, regions, adjacency_matrix, faces, indices, head_meshes, meshes_to_pool, ids, num_faces, self.target_size, faces_non_iterate, self.device)
            meshes_not_poolable = torch.cat((meshes_not_poolable, ret_glob_regions[-1])).type(torch.int)
            if len(ret_glob_regions)>1:
                adj_face_1_glob, adj_face_2_glob, regions_glob, faces_to_compress, erased_iter, idx_glob, indices, _, _ = ret_glob_regions
                
                idx0,idx1 = torch.where(adj_face_1_glob > adj_face_2_glob)
                adj_face_1_glob[idx0,idx1], adj_face_2_glob[idx0,idx1] = adj_face_2_glob[idx0,idx1], adj_face_1_glob[idx0,idx1]
                regions_glob[idx0,0,idx1,-1], regions_glob[idx0,1,idx1,-1] = regions_glob[idx0,1,idx1,-1], regions_glob[idx0,0,idx1,-1]

                adjacency_matrix, erased_iter, inverse_inds1, inverse_inds2 = mesh_pool.update_adj_matrix_parallel(adjacency_matrix, indices, regions_glob, adj_face_1_glob, adj_face_2_glob, erased_iter, regions_glob.shape[2], self.device)
                # Now we have all the faces that are going to be removed by our operation
                # Here we update the features of the adj_faces
                torch.cuda.empty_cache()
                
                features = mesh_pool.update_features_parallel(features, indices, regions_glob, adj_face_1_glob, adj_face_2_glob, regions_glob.shape[2], num_channels, idx_glob, erased_iter, self.choice_points, self.device)
               
                faces, adjacency_matrix, features, erased_iter = mesh_pool.update_faces(data, faces, indices, regions_glob, faces_to_compress, adjacency_matrix, features, erased_iter, idx_glob, adj_face_1_glob, adj_face_2_glob, inverse_inds1, inverse_inds2)
                # Update list of erased faces
                erased = torch.cat((erased, erased_iter), dim=0).unique(dim=0).type(torch.int)
                erased = erased[erased[:,1]!=-1]
                
                erased = torch.cat((erased,data.erased)).unique(dim=0)
                erased = erased[erased[:,1]!=-1]
                ring_k_old = ring_k.clone().to(self.device)
                data.compute_ring_k_batch(4)
                ring_k = data.ring_k

                # Update of the tensors used for unpooling
                erased_pool = torch.cat((erased_pool,erased_iter),dim=0)
                adj_face_1_pool = torch.cat((adj_face_1_pool, adj_face_1_glob.transpose(0,1)),dim=0)
                adj_face_2_pool = torch.cat((adj_face_2_pool, adj_face_2_glob.transpose(0,1)),dim=0)
                reg1_pool = torch.cat((reg1_pool, regions_glob[:,0,:,-1].transpose(0,1)),dim=0)
                reg2_pool = torch.cat((reg2_pool, regions_glob[:,1,:,-1].transpose(0,1)),dim=0)
                indices_iter = torch.cat((indices_iter, (torch.ones(inverse_inds1.shape[0]) * cpt).type(torch.int).to(self.device)),dim=0)
                indices_iter1 = torch.cat((indices_iter1, (torch.ones(inverse_inds2.shape[0]) * cpt).type(torch.int).to(self.device)),dim=0)
                indices_iter2 = torch.cat((indices_iter2, (torch.ones(adj_face_1_glob.shape[1]) * cpt).type(torch.int).to(self.device)),dim=0)
                indices_pool = torch.cat((indices_pool, indices),dim=0)
                inverse_indices1 = torch.cat((inverse_indices1, inverse_inds1),dim=0)
                inverse_indices2 = torch.cat((inverse_indices2, inverse_inds2),dim=0)
                cpt+=1

                # compute updated weights
                if self.choice_points:
                    idx[meshes_to_pool], sorted[meshes_to_pool] = mesh_pool.update_weights_points(features[meshes_to_pool], ring_k_old[meshes_to_pool], ring_k[meshes_to_pool], meshes_to_pool.shape[0], features.shape[1], num_channels, sorted[meshes_to_pool], idx[meshes_to_pool], erased[torch.isin(erased[:,0], meshes_to_pool)], self.device, 0)
                else:
                    idx[meshes_to_pool], sorted[meshes_to_pool] = mesh_pool.update_weights(features[meshes_to_pool], ring_k_old[meshes_to_pool], ring_k[meshes_to_pool], meshes_to_pool.shape[0], features.shape[1], num_channels, sorted[meshes_to_pool], idx[meshes_to_pool], erased[torch.isin(erased[:,0], meshes_to_pool)], self.device, 0)
                torch.cuda.empty_cache()

            unique,counts = erased[:,0].unique(return_counts = True)
            meshes_to_pool = unique[(features.shape[1] - counts) > self.target_size]
            num_faces = (features.shape[1] - counts)[meshes_to_pool]

            mask = torch.isin(meshes_to_pool,meshes_not_poolable,invert=True)
            meshes_to_pool = meshes_to_pool[mask]
            num_faces = num_faces[mask]
        
        if erased_pool.size(0)>0:
            unpool_material = [adjacency_matrix[erased_pool[:,0],erased_pool[:,1]], adj_face_1_pool, adj_face_2_pool, reg1_pool, reg2_pool, indices_pool, indices_iter, indices_iter1, indices_iter2, erased_pool, inverse_indices1, inverse_indices2, old_faces]
            data.unpool_material.append(unpool_material)

        unique,counts = erased[:,0].unique(return_counts = True)
        num_faces = (features.shape[1] - counts)

        # Update the data to continue the execution of the network
        data[0] = faces
        data[1] = features.requires_grad_()
        data[2] = adjacency_matrix
        data[3] = ring_k

        data, erased = utils.remove_zeros(data, num_faces.max(), erased, self.device, self.choice_points)#torch.device('cpu'))
        data[-1] = erased
        
        # Recompute the K neighborhoods of the updated meshes for all the faces
        data.compute_ring_k_batch(self.K)        
        return data

class MeshUnPool_memory(nn.Module):
    """
    Face unpooling of a mesh by face colapse.
    K           : size of the convolution kernel that we need to find at the end of the pooling
    points      : Is the operation applied on the points themselves or on features of the faces
    """

    def __init__(self, K, points = False):
        super(MeshUnPool_memory, self).__init__()
        self.K = K
        self.choice_points = points

    def forward(self, data):
        unpool_material = data.unpool_material.pop(-1)
        if len(unpool_material)>0:
            self.device = data[1].device

            data[1], data[3] = data[1].clone().to(self.device), data[3].clone().to(self.device)
            new_faces = unpool_material.pop(-1)

            # Restore connectivity
            data, faces_to_restore = mesh_unpool.restore_connectivity_memory(data, unpool_material, self.choice_points)
            
            # Create new features for restored faces
            data = mesh_unpool.update_features_of_restored_faces(data,faces_to_restore)
            data.faces = new_faces.type(torch.long)

            data.compute_ring_k_batch(self.K)
        
        return data

class ReLU(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self):
        super(ReLU, self).__init__()
        self.ReLU = nn.GELU()

    def forward(self, mesh):
        mesh[1] = self.ReLU(mesh[1])
        return mesh

class MeshLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, mesh):
        erased = mesh[-1].type(torch.int)

        mesh[1] = self.conv1d(mesh[1].transpose(1,2)).transpose(1,2)
        mesh[1][erased[:,0],erased[:,1]] = 0

        return mesh