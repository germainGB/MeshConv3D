import torch
import torch.nn as nn
import numpy as np
import utils
import time

def restore_connectivity_memory(data, unpool_material, choice_points):
    device = data[2].device

    features = data[1]
    adjacency_matrix = data[2]
    erased = data[-1]
    N = adjacency_matrix.shape[0]

    adjacency_erased, adj_face_1_pool, adj_face_2_pool, reg1_pool, reg2_pool, indices_pool, indices_iter, indices_iter1, indices_iter2, erased_pool, inverse_indices1, inverse_indices2 = unpool_material
    unique, counts_temp = erased_pool[erased_pool[:,1] != -1].unique(dim=0)[:,0].unique(return_counts=True)
    unique_2,counts_2_temp = erased[:,0].unique(return_counts=True)
    counts = torch.zeros(adjacency_matrix.size(0)).type(torch.long).to(device)
    counts[unique] = counts_temp
    counts_2 = torch.zeros(adjacency_matrix.size(0)).type(torch.long).to(device)
    counts_2[unique_2] = counts_2_temp
    new_shape = adjacency_matrix.shape[1]+(counts - counts_2).max() + 1

    new_adjacency_matrix = torch.ones(N,new_shape,3).type(torch.int).to(device) * -1
    if choice_points:
        new_features = torch.zeros(N,new_shape,features.shape[2], features.shape[3]).type(features.type()).to(device)
    else:
        new_features = torch.zeros(N,new_shape,features.shape[2]).type(features.type()).to(device)

    new_erased = torch.Tensor().type(torch.int).to(device)
    faces_to_restore = torch.Tensor().type(torch.int).to(device)

    for i in range(N):
        erased_i = erased_pool[erased_pool[:,0] == i]

        idx_erased, = torch.where(erased_i[:,1] != -1)

        map = torch.zeros(new_shape).type(torch.int).to(device)
        map[erased_i[idx_erased,1]] = 1
        map2 = torch.ones(adjacency_matrix.shape[1]).type(torch.int).to(device) * -1

        idx_new, = torch.where(map == 0)
        temp_shape = adjacency_matrix.shape[1] - erased[erased[:,0] == i].shape[0]

        idx_new = idx_new[:temp_shape]
        # idx_new[-1] = -1

        mask = torch.ones(adjacency_matrix.shape[1],dtype=bool)
        mask[erased[erased[:,0] == i][:,1]] = False
        map2[mask] = torch.arange(temp_shape).type(torch.int).to(device)
        adja_mat_i = map2[adjacency_matrix[i,mask]]

        new_adjacency_matrix[i,idx_new] = (idx_new[adja_mat_i]).type(torch.int)
        new_features[i,idx_new] = features[i, mask]

        # new erased
        new_erased_i = torch.arange(new_shape).type(torch.int).to(device)
        mask = torch.ones(new_erased_i.shape[0],dtype=bool)
        mask[idx_new] = False
        mask[erased_i[:,1]] = False
        new_erased_i = new_erased_i[mask]
        new_erased_i = torch.cat((new_erased_i,(torch.ones(1).to(device)*new_shape - 1).to(device).type(torch.int)))

        faces_to_restore_i = torch.arange(new_shape).unsqueeze(1).type(torch.int).to(device)
        mask = torch.ones(new_shape, dtype=bool)
        mask[new_erased_i] = False
        mask[idx_new] = False
        faces_to_restore_i = faces_to_restore_i[mask]
        faces_to_restore = torch.cat((faces_to_restore,torch.cat(((torch.ones(faces_to_restore_i.shape[0],1).to(device).type(torch.int) * i), faces_to_restore_i), dim=1)), dim=0)

        new_erased = torch.cat((new_erased,torch.cat(((torch.ones(new_erased_i.shape[0],1).to(device).type(torch.int) * i), new_erased_i.unsqueeze(1)), dim=1)), dim=0)

        adjacency_i = adjacency_erased[erased_pool[:,0] == i]

        new_adjacency_matrix[i,erased_i[:,1]] = (adjacency_i).type(torch.int)
        new_adjacency_matrix[i,-1] = -1
        
    max = indices_iter.max()
    for j in range(max+1):
        k = max - j
        idx2, = torch.where(indices_iter2 == k)

        idx, = torch.where(indices_iter1 == k)
        idx10, idx11, idx12 = inverse_indices2[idx][:,0], inverse_indices2[idx][:,1], inverse_indices2[idx][:,2]
        new_adjacency_matrix[indices_pool[idx2][idx11], adj_face_2_pool[idx2][idx11,idx10], idx12] = (reg2_pool[idx2][idx11,idx10]).type(torch.int)

        idx, = torch.where(indices_iter == k)
        idx10, idx11, idx12 = inverse_indices1[idx][:,0], inverse_indices1[idx][:,1], inverse_indices1[idx][:,2]
        new_adjacency_matrix[indices_pool[idx2][idx11], adj_face_1_pool[idx2][idx11,idx10], idx12] = (reg1_pool[idx2][idx11,idx10]).type(torch.int)

    data[1] = new_features
    data[2] = new_adjacency_matrix
    data[-1] = new_erased.unique(dim=0)

    return data, faces_to_restore

def update_features_of_restored_faces(data,faces_to_restore):
    adjacency_matrix = data[2]
    features = data[1]

    for i in faces_to_restore[:,0].unique():
        faces_to_restore_i = faces_to_restore[faces_to_restore[:,0] == i]
        faces_to_restore_neighbors = adjacency_matrix[faces_to_restore_i[:,0],faces_to_restore_i[:,1]]
        mem = -1
        mem_1 = -2
        cpt=0
        
        while faces_to_restore_i.shape[0]>0:
            idx0,idx1 = torch.where(torch.isin(faces_to_restore_neighbors, faces_to_restore_i[:,1]))

            unique,counts = idx0.unique(return_counts=True)
            
            idx_full = torch.arange(faces_to_restore_i.shape[0]).to(data.device)
            mask = torch.ones(idx_full.shape[0],dtype=bool)
            mask[unique] = False
            idx_full = idx_full[mask]
            
            idx101,_ = torch.where(faces_to_restore_neighbors[idx_full] != -1)
            unique2,counts2 = idx101.unique(return_counts = True)
            features[i, faces_to_restore_i[idx_full,1]] = torch.sum(features[i,faces_to_restore_neighbors[idx_full]],dim=1) / counts2.unsqueeze(1)

            idx150,_ = torch.where(faces_to_restore_neighbors[unique] == -1)
            unique2,counts2 = idx150.unique(return_counts = True)
            counts[unique2] += counts2
            idx = unique[counts>1]
            idx = idx[torch.isin(idx,idx_full,invert=True)]

            mask = torch.ones(faces_to_restore_i.shape[0],dtype=bool)
            mask[idx] = False
            mask[idx_full] = False

            faces_restored_i = faces_to_restore_i[mask]
            faces_restored_neighbors = faces_to_restore_neighbors[mask]

            idx1,idx2 = torch.where(torch.isin(faces_restored_neighbors, faces_to_restore_i[:,1]))

            faces_to_restore_i = faces_to_restore_i[idx]
            faces_to_restore_neighbors = faces_to_restore_neighbors[idx]
            
            idx = torch.arange(3).unsqueeze(0).repeat(faces_restored_i.shape[0],1)
            mask = torch.ones(idx.numel(),dtype=bool).view(-1,3)
            mask[idx1,idx2] = False
            idx = idx[mask]
            idx = idx.view(-1,2)
            
            idx101,_ = torch.where(faces_restored_neighbors[torch.arange(faces_restored_i.shape[0])[:,None], idx] != -1)
            unique2,counts2 = idx101.unique(return_counts = True)
            
            features[faces_restored_i[:,0],faces_restored_i[:,1]] = torch.sum(features[i, faces_restored_neighbors[torch.arange(faces_restored_i.shape[0])[:,None], idx]],dim=1) / counts2.unsqueeze(1)

            if mem == faces_to_restore_i.shape[0]:
                idx = unique[counts == 2]
                idx3,_ = torch.where(faces_to_restore_neighbors[idx] == -1)
                if idx.size(0) != idx3.unique().size(0):
                    mask = torch.ones(idx.size(0), dtype=bool)
                    mask[idx3] = False
                    idx = idx[mask]
                idx1,idx2 = torch.where((torch.isin(faces_to_restore_neighbors[idx],faces_to_restore_i[:,1],invert=True)) & (faces_to_restore_neighbors[idx] != -1))
                features[faces_to_restore_i[idx,0],faces_to_restore_i[idx,1]] = features[faces_to_restore_i[idx,0], faces_to_restore_neighbors[idx,idx2]]
                mask = torch.ones(faces_to_restore_i.shape[0],dtype=bool)
                mask[idx] = False

                faces_to_restore_i = faces_to_restore_i[mask]
                faces_to_restore_neighbors = faces_to_restore_neighbors[mask]
                mem_mem = mem
                mem = faces_to_restore_i.shape[0]
            else: 
                mem_mem = mem
                mem = faces_to_restore_i.shape[0]

            if mem_1 == faces_to_restore_i.shape[0]:
                features[faces_to_restore_i[:,0],faces_to_restore_i[:,1]] = torch.randn(faces_to_restore_i.size(0),features.size(2)).to(features.device).type(features.dtype)

                faces_to_restore_i = torch.Tensor()
                faces_to_restore_neighbors = torch.Tensor()
            else :
                mem_1 = mem_mem

            cpt+=1

    data[1] = features
    return data