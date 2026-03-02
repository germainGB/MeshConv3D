import torch
import numpy as np
import torch.nn as nn
import time
import utils

def compute_weights_points(features, ring_k, num_meshes, num_faces, num_channels, erased):
    ring_k_1 = ring_k[:,:,:4].unsqueeze(3).unsqueeze(4).clone().type(torch.int)
    idx0, idx1, idx2, _, _ = torch.where(ring_k_1 == -1)
    ring_k_1[idx0, idx1, idx2] = ring_k_1[idx0, idx1, 0]
    # print(features[9,0])

    ring_k_1 = ring_k_1.expand(num_meshes, num_faces, -1, num_channels, 3)
    # features of the three faces in the 1-ring neighborhood of each face of the mesh and the center face itself
    neighbor_fea = features[
        torch.arange(num_meshes)[:, None, None, None, None],
        ring_k_1,
        torch.arange(num_channels)[None, None, None, :, None],
        torch.arange(3)[None, None, None, None, :]
    ]

    # Compute the weights of the faces
    w = torch.norm(features.unsqueeze(2) - neighbor_fea, p=2, dim=[3,4])
    w = torch.sum(w, dim=2)

    w[erased[:,0],erased[:,1]] = float('inf')

    # Select the faces with the lowest weights

    sorted, idx = torch.sort(w, dim=1)

    return idx, sorted

def update_weights_points(features, ring_k, ring_k_new, num_meshes, num_faces2, num_channels,sorted, idx, erased_regions, device, choice=0):
    """
    This function updates the weights of the faces whose 1-ring neighborhood has been modified during the pooling operation
    """
    # Identify all the faces whose 1-ring neighborhood has been modified
    idx_modified, idx_modified_2,_ = torch.where(ring_k[:,:,:4] != ring_k_new)
    idx_modified = torch.cat((idx_modified.view(-1,1), idx_modified_2.view(-1,1)), dim=1)
    if choice == 1:
        erased_regions = erased_regions.clone().to(device).type(torch.int)
        erased_regions[:,0] = 0
    # idx_modified = torch.cat((idx_modified, erased_regions),dim=0)
    unique = idx_modified.unique(dim=0)

    #remove faces -1
    idx_44, = torch.where(unique[:,1]==-1)
    mask = torch.ones(unique.shape[0], dtype = bool)
    mask[idx_44] = False
    unique = unique[mask]

    idx_keep = torch.arange(unique.shape[0])#[mask]
    
    # We recompute the weights of the identified faces (setting inf for removed faces)
    w = (torch.ones(unique.shape[0]) * float('inf')).to(device)
    temp_ring = ring_k_new[unique[idx_keep,0], unique[idx_keep,1],:4].clone().type(torch.int)
    idx0, idx1 = torch.where(temp_ring == -1)
    temp_ring[idx0, idx1] = temp_ring[idx0, 0]
    neighbor_fea = features[unique[idx_keep,0][:,None],temp_ring]

    temp_norm = torch.norm(features[unique[idx_keep,0], unique[idx_keep,1]].unsqueeze(1) - neighbor_fea, p=2, dim=[2,3])
    temp_norm = torch.sum(temp_norm, dim=1)
    w[idx_keep] = temp_norm

    map = (torch.ones(erased_regions[:,0].max()+1)*-1).to(device).type(torch.int)
    map[erased_regions[:,0].unique()] = torch.arange(erased_regions[:,0].unique().shape[0]).to(device).type(torch.int)
    
    # We insert the updated weights and update the sorting of the weights
    unsorted = sorted.gather(1, idx.argsort(1))
    unsorted[unique[:,0],unique[:,1]] = w
    unsorted[map[erased_regions[:,0]],erased_regions[:,1]] = float('inf')

    # print(sorted[3,idx_22])

    sorted, idx = torch.sort(unsorted, dim=1)
    # idx = idx[torch.arange(num_meshes)[:,None], idx_new]

    return idx, sorted

def compute_weights(features, ring_k, num_meshes, num_faces, num_channels,erased, descending=False, unsorted=False):
    """
    Computes the initial weights of each face of each mesh
    """
    # Gather features at face neighbors
    tic = time.time()
    neighbor_fea = []
    for i in range(4):
        neighbor_fea.append(features[torch.arange(num_meshes)[:,None],ring_k[:,:,i]])
    
    neighbor_fea = torch.stack(neighbor_fea,2)

    # Compute the weights of the faces
    w = torch.norm(features.unsqueeze(2) - neighbor_fea, p=2, dim=3)
    w = torch.sum(w, dim=2)

    if descending:
        w[erased[:,0],erased[:,1]] = - float('inf')
        if unsorted: return [],w
        # Select the faces with the lowest weights
        sorted, idx = torch.sort(w, dim=1, descending=True)
    else:
        w[erased[:,0],erased[:,1]] = float('inf')
        if unsorted: return [],w
        # Select the faces with the lowest weights
        sorted, idx = torch.sort(w, dim=1)

    return idx, sorted

def update_weights(features, ring_k, ring_k_new, num_meshes, num_faces2, num_channels,sorted, idx, erased_regions, device, choice=0):
    """
    This function updates the weights of the faces whose 1-ring neighborhood has been modified during the pooling operation
    """
    # Identify all the faces whose 1-ring neighborhood has been modified
    idx_modified, idx_modified_2,_ = torch.where(ring_k[:,:,:4] != ring_k_new)
    idx_modified = torch.cat((idx_modified.view(-1,1), idx_modified_2.view(-1,1)), dim=1)
    if choice == 1:
        erased_regions = erased_regions.clone().to(device).type(torch.int)
        erased_regions[:,0] = 0
    unique = idx_modified.unique(dim=0)

    #remove faces -1
    idx_44, = torch.where(unique[:,1]==-1)
    mask = torch.ones(unique.shape[0], dtype = bool)
    mask[idx_44] = False
    unique = unique[mask]

    # Here we remove from that list the faces that have been removed
    idx_keep = torch.arange(unique.shape[0])#[mask]
    
    # We recompute the weights of the identified faces (setting inf for removed faces)
    w = (torch.ones(unique.shape[0]) * float('inf')).to(device)
    temp_ring = ring_k_new[unique[idx_keep,0], unique[idx_keep,1],:4].clone().type(torch.int)
    idx0, idx1 = torch.where(temp_ring == -1)
    temp_ring[idx0, idx1] = temp_ring[idx0, 0]
    neighbor_fea = features[unique[idx_keep,0][:,None],temp_ring]

    temp_norm = torch.norm(features[unique[idx_keep,0], unique[idx_keep,1]].unsqueeze(1) - neighbor_fea, p=2, dim=2)
    temp_norm = torch.sum(temp_norm, dim=1)
    w[idx_keep] = temp_norm

    map = (torch.ones(erased_regions[:,0].max()+1)*-1).to(device).type(torch.int)
    map[erased_regions[:,0].unique()] = torch.arange(erased_regions[:,0].unique().shape[0]).to(device).type(torch.int)
    
    # We insert the updated weights and update the sorting of the weights
    unsorted = sorted.gather(1, idx.argsort(1))
    unsorted[unique[:,0],unique[:,1]] = w
    unsorted[map[erased_regions[:,0]],erased_regions[:,1]] = float('inf')
    sorted, idx = torch.sort(unsorted, dim=1)

    return idx, sorted

def find_faces_staying(adjacency_matrix, indices, regions, num_meshes, idx_neighbor_face, faces_non_iterate,device):
    """
    Function that finds the adj_faces of regions and solves some of the special cases
    """
    tac = time.time()
    # Faces to be removed (in addition to the 3 regions and the center face)
    ids = torch.Tensor(0,2).type(torch.int).to(device)
    adj_face_1 = (torch.ones(num_meshes,3) * -1).type(torch.long).to(device)
    adj_face_2 = (torch.ones(num_meshes,3) * -1).type(torch.long).to(device)
    # Find neighbors of the three regions
    idx1, idx100, idx2 = torch.where(adjacency_matrix[indices[:,None], regions[idx_neighbor_face-1, 0, :, 1:4][None,:]].squeeze(0) != regions[idx_neighbor_face-1,0,:,0].unsqueeze(1).unsqueeze(2))
    unique,counts=idx1.unique(return_counts=True)
    # print(unique, counts)
    # We remove meshes where the region selected is just 2 faces with all three edges linked together
    mask = torch.ones(num_meshes,dtype=bool)
    mask[unique] = False
    three_connected = torch.arange(num_meshes)[mask].to(device)
    # As it is just 2 faces we do not want to iterate on them
    faces_non_iterate = torch.cat((faces_non_iterate, torch.cat((torch.arange(3).repeat(three_connected.shape[0]).view(-1,1).to(device), three_connected.view(-1,1).repeat(1,3).view(-1,1)), dim=1)),dim=0)
    adj_face_1[three_connected] = regions[idx_neighbor_face-1,0,three_connected,0].view(-1,1).repeat(1,3)
    adj_face_2[three_connected] = regions[idx_neighbor_face-1,0,three_connected,0].view(-1,1).repeat(1,3)
    # Here we find meshes where one of the regions is connected twice to the same face
    mesh_wrong = unique[counts!=6]
    meshes_case_2 = torch.Tensor().to(device)
    tic = time.time()
    if mesh_wrong.shape[0]>0:
        # Find meshes where the center face is connected twice to one of the regions
        idx3_1, idx3_2, _ = torch.where(regions[0,0,mesh_wrong,1:4].unsqueeze(1) == regions[0,0,mesh_wrong,1:4].unsqueeze(2))
        unique,counts = idx3_1[regions[0,0,mesh_wrong[idx3_1],idx3_2+1] != -1].unique(return_counts=True)
        # print(idx3_1, idx3_2)
        id_keep, = torch.where(torch.isin(idx3_1, unique[counts>1]))
        # _,id_keep = torch.where(idx3_1 == unique[counts>1].unsqueeze(1))
        idx3_1 = mesh_wrong[idx3_1[id_keep]]
        idx3_2 = idx3_2[id_keep]
        unique,counts = idx3_1.unique(return_counts=True)
        meshes_case_1 = unique[counts>3]
        meshes_case_2 = meshes_case_1.clone().type(torch.int)
        if meshes_case_1.shape[0]>0:
            # Find the 2 problematic faces
            _,idx4 = torch.where(idx3_1 == meshes_case_1.unsqueeze(1))
            idx3_2 = idx3_2[idx4].view(meshes_case_1.shape[0],-1)
            idx_region_change = torch.Tensor(meshes_case_1.shape[0],2).type(torch.int).to(device)
            idx_region_unchange = torch.Tensor(meshes_case_1.shape[0],1).type(torch.int).to(device)
            for i in range(3):
                idx_temp,_ = torch.where(idx3_2 == i)
                unique, counts = idx_temp.unique(return_counts=True)
                ids_temp = [0,1,2]
                ids_temp.pop(i)
                idx_region_change[unique[counts==1]] = torch.Tensor(ids_temp).type(torch.int).to(device)
                idx_region_unchange[unique[counts==1]] = i

            _,idx5 = torch.where(adjacency_matrix[indices[meshes_case_1], regions[idx_region_change[:,0],0,meshes_case_1,idx_region_change[:,0]+1]] != regions[0,0,meshes_case_1,0].unsqueeze(1))
            # Find the faces adjacent to the adj_faces and that will later be deleted
            new_regs = adjacency_matrix[indices[meshes_case_1], regions[idx_region_change[:,0],0,meshes_case_1,idx_region_change[:,0]+1], idx5]
            reg1 = regions[idx_region_unchange.view(-1),0,meshes_case_1,idx_region_change[:,0]+1]

            id_remove, = torch.where(new_regs == -1)
            mesh_change,_ = torch.where(regions[0, 0, meshes_case_1[id_remove,None], idx_region_unchange[id_remove]+1] != -1)

            _,_,idx10 = torch.where(adjacency_matrix[indices[meshes_case_1[id_remove[mesh_change],None]], regions[0, 0, meshes_case_1[id_remove[mesh_change],None], idx_region_unchange[id_remove[mesh_change]]+1]].squeeze(0) != regions[0, 0, meshes_case_1[id_remove[mesh_change],None], 0].unsqueeze(2))
            idx10 = idx10.view(-1, 2)

            adj_face_1[meshes_case_1[id_remove[mesh_change]][:,None], idx_region_unchange[id_remove[mesh_change]]] = adjacency_matrix[indices[meshes_case_1[id_remove[mesh_change],None]], regions[0, 0, meshes_case_1[id_remove[mesh_change],None], idx_region_unchange[id_remove[mesh_change]]+1]].reshape(mesh_change.shape[0],3)[torch.arange(mesh_change.shape[0]),idx10[:,0]].unsqueeze(1)
            adj_face_2[meshes_case_1[id_remove[mesh_change]][:,None], idx_region_unchange[id_remove[mesh_change]]] = adjacency_matrix[indices[meshes_case_1[id_remove[mesh_change],None]], regions[0, 0, meshes_case_1[id_remove[mesh_change],None], idx_region_unchange[id_remove[mesh_change]]+1]].reshape(mesh_change.shape[0],3)[torch.arange(mesh_change.shape[0]),idx10[:,1]].unsqueeze(1)

            mask = torch.ones(meshes_case_1.numel(), dtype=torch.bool)
            mask[id_remove] = False
            meshes_case_1, idx_region_change, idx_region_unchange, new_regs, reg1 = meshes_case_1[mask], idx_region_change[mask], idx_region_unchange[mask], new_regs[mask], reg1[mask]
            # Verify that there are not regions
            new_regs,temp = verif_new_faces_are_not_regions(adjacency_matrix, indices, adj_face_1.T, new_regs, reg1, regions, meshes_case_1, 0, torch.arange(meshes_case_1.shape[0]).type(torch.int).to(device), 1, torch.Tensor(), ids)
            old_regs = regions[:,0,meshes_case_1,0].T.view(-1,3)
            old_regs[torch.arange(meshes_case_1.shape[0])[:,None],idx_region_change] = temp.type(torch.long).view(-1,1).expand(-1,2) #regions[idx_region_change[:,0],0,meshes_case_1,idx_region_change[:,0]+1].clone().type(torch.long).view(-1,1).expand(-1,2)
            # Update the regions
            regions[idx_region_change,:,meshes_case_1[:,None],idx_region_change+1] = new_regs.view(-1,1,1).expand(-1,2,2)
            regions[idx_region_change,:,meshes_case_1[:,None],-1] = new_regs.view(-1,1,1).expand(-1,2,1)
            regions[idx_region_unchange.view(-1),:,meshes_case_1,idx_region_change[:,0]+1] = new_regs.view(-1,1).expand(-1,2)
            # Find the new distinct adj_faces (neighbors of the new regs)
            idx7,_,idx6 = torch.where(adjacency_matrix[indices[meshes_case_1[:,None]], regions[torch.arange(3), 0, meshes_case_1[:,None], torch.arange(3)+1]].squeeze(0) != old_regs.unsqueeze(2)) # mettre meshes case 2 ici?

            idx0_0, idx1_0 = torch.where((regions[torch.arange(3), 0, meshes_case_1[:,None], torch.arange(3)+1] == -1))
            mask = torch.ones(idx6.numel(), dtype=torch.bool)
            prec, prec_prec = -1, -1
            for i in range(idx0_0.shape[0]):
                offset = idx7[idx7 < idx0_0[i]].shape[0] #- i
                if prec == idx0_0[i]: offset+=1
                if prec_prec == idx0_0[i]: offset+=1
                prec_prec, prec = prec, idx0_0[i]

                mask[2*idx1_0[i] + offset] = False

                # mask[2*idx1_0[i] + 6 * idx0_0[i] + i] = False
            unique1,counts1 = idx7[mask].unique(return_counts = True)

            idx6 = idx6[mask]

            idx6 = idx6.view(-1, 3, 2)

            adj_face_1[meshes_case_1] = adjacency_matrix[indices[meshes_case_1[:,None]], regions[torch.arange(3), 0, meshes_case_1[:,None], torch.arange(3)+1], idx6[:,:,0]].squeeze(0)
            adj_face_2[meshes_case_1] = adjacency_matrix[indices[meshes_case_1[:,None]], regions[torch.arange(3), 0, meshes_case_1[:,None], torch.arange(3)+1], idx6[:,:,1]].squeeze(0)
    time1 = time.time() - tic

    idx3, = torch.where(torch.isin(idx1, meshes_case_2))

    mask = torch.ones(idx1.numel(), dtype=torch.bool)
    mask[idx3] = False

    tic = time.time()
    temp = torch.cat((idx1.unsqueeze(1), idx100.unsqueeze(1)), dim=1)
    unique, inverse, counts = temp.unique(dim=0, return_counts=True,return_inverse=True)
    temp2, = torch.where(counts>2)
    idx_mask, = torch.where(torch.isin(inverse,temp2))
    idx_mask = idx_mask[::3]
    mask[idx_mask] = False

    unique1,counts1 = idx1[mask].unique(return_counts = True)
    # print(unique1[counts1>6])
    idx1 = idx1[mask].unique()
    idx = idx2[mask].view(-1, 3, 2)
    # The adjacency faces that will be connected after the first edge collapse
    adj_face_1[idx1] = adjacency_matrix[indices[idx1[:,None]], regions[idx_neighbor_face-1, 0, idx1, 1:4][None,:], idx[:,:,0]].squeeze(0)
    adj_face_2[idx1] = adjacency_matrix[indices[idx1[:,None]], regions[idx_neighbor_face-1, 0, idx1, 1:4][None,:], idx[:,:,1]].squeeze(0)
    tic=time.time()
    faces_non_iterate = (check_zone_is_not_manifold(adjacency_matrix, indices, adj_face_1.T, adj_face_2.T, regions, ids, faces_non_iterate, device))
    faces_non_iterate = find_and_solve_holes(adj_face_1, adj_face_2, regions, idx_neighbor_face, faces_non_iterate)

    # Find regions where the adj_faces are equal and iterate we find 2 distinct adj_faces
    adj_face_1, adj_face_2, regions, ids, faces_non_iterate,_ = find_and_resolve_meshes_with_same_adj_faces(adjacency_matrix, indices, adj_face_1.T, adj_face_2.T, regions, ids, faces_non_iterate, device)
    faces_non_iterate = find_and_solve_holes(adj_face_1.T, adj_face_2.T, regions, idx_neighbor_face, faces_non_iterate)
    faces_non_iterate = (check_zone_is_not_manifold(adjacency_matrix, indices, adj_face_1, adj_face_2, regions, ids, faces_non_iterate, device))
    time3 = time.time() - tic
    time0 = time.time() - tac

    return adj_face_1, adj_face_2, regions, ids, faces_non_iterate

def find_and_solve_holes(adj_face_1, adj_face_2, regions, idx_neighbor_face, faces_non_iterate):
    # Treat the holes in the structure
    idx0_0, idx1_0 = torch.where((regions[idx_neighbor_face-1, 0, :, 1:4] == -1))
    idx0_1, idx1_1 = torch.where(adj_face_1 == -1)
    idx0_2, idx1_2 = torch.where(adj_face_2 == -1)

    idx0 = torch.cat((idx0_0, idx0_1, idx0_2))
    idx1 = torch.cat((idx1_0, idx1_1, idx1_2))

    if idx0.shape[0]>0:
        faces_non_iterate = torch.cat((faces_non_iterate, torch.cat((idx1.view(-1,1), idx0.view(-1,1)), dim=1)),dim=0).unique(dim=0)
    return faces_non_iterate

def find_adj_faces_adjacent_region2(adjacency_matrix, indices, adj_face_1, adj_face_2, regions, ids, faces_non_iterate, device):
    """
    This function treats the cases when one of the adj_faces of a region of a mesh is in the removed zone
    """
    adj_face_1_old = adj_face_1.clone().type(torch.long)
    adj_face_2_old = adj_face_2.clone().type(torch.long)
    regions_old = regions.clone().type(torch.long)
    num_meshes = indices.shape[0]

    # Remove faces_non_iterate so that we don't end up having a bug
    mask = torch.ones(adj_face_1.numel(),dtype=bool).view(3,-1)
    mask[faces_non_iterate[:,0],faces_non_iterate[:,1]] = False
    diff = torch.cumsum(mask, dim=1).to(device)
    diff[1,:] = diff[1,:] + diff[0,-1]
    diff[2,:] = diff[2,:] + diff[1,-1]
    diff[faces_non_iterate[:,0],faces_non_iterate[:,1]] = -1
    idx10, idx11 = torch.where(diff!=-1)
    idxs15 = torch.cat((diff[mask].unsqueeze(1), idx10.unsqueeze(1), idx11.unsqueeze(1)),dim=1)

    # Find adj_faces that are actually regions[0:4] or adj_faces that are regions[-1]
    idx1,idx1_3 = torch.where(regions[:,0,:,1:4][mask] == adj_face_1[mask].unsqueeze(1))
    idx1_1, idx1_2 = idxs15[idx1,1], idxs15[idx1,2]

    idx2,idx2_3 = torch.where(regions[:,1,:,1:4][mask] == adj_face_2[mask].unsqueeze(1))
    idx2_1, idx2_2 = idxs15[idx2,1], idxs15[idx2,2]

    # Finally find the new adj_faces that are not regions anymore
    adj_face_1, adj_face_2, regions, ids, faces_non_iterate = find_new_faces_adj_regions(adjacency_matrix, indices, adj_face_1, adj_face_2, adj_face_1_old, adj_face_2_old, regions, regions_old, ids, faces_non_iterate, idx1_1, idx1_2, idx1_3, 0)
    adj_face_2, adj_face_1, regions, ids, faces_non_iterate = find_new_faces_adj_regions(adjacency_matrix, indices, adj_face_2, adj_face_1, adj_face_2_old, adj_face_1_old, regions, regions_old, ids, faces_non_iterate, idx2_1, idx2_2, idx2_3, 1)
    faces_non_iterate = find_and_solve_holes(adj_face_1.T, adj_face_2.T, regions, 0, faces_non_iterate)
    
    # Find regions where the adj_faces are equal and iterate we find 2 distinct adj_faces
    adj_face_1, adj_face_2, regions, ids, faces_non_iterate, _ = find_and_resolve_meshes_with_same_adj_faces(adjacency_matrix, indices, adj_face_1, adj_face_2, regions, ids, faces_non_iterate, device)
    faces_non_iterate = find_and_solve_holes(adj_face_1.T, adj_face_2.T, regions, 0, faces_non_iterate)

    change = True
    # Iterate while we make modifications and we find special cases

    while change:
        change = False

        idx18, = torch.where(((adj_face_1[faces_non_iterate[:,0], faces_non_iterate[:,1]] == -1) & (adj_face_2[faces_non_iterate[:,0], faces_non_iterate[:,1]] != -1)) |
                    ((adj_face_2[faces_non_iterate[:,0], faces_non_iterate[:,1]] == -1) & (adj_face_1[faces_non_iterate[:,0], faces_non_iterate[:,1]] != -1)))
        mask = torch.ones(faces_non_iterate.shape[0],dtype=bool)
        mask[idx18] = False
        faces_non_iterate_2 = faces_non_iterate[mask]

        mask2 = torch.ones(adj_face_1.numel(),dtype=bool).view(3,-1)
        mask2[faces_non_iterate_2[:,0],faces_non_iterate_2[:,1]] = False
        diff = torch.cumsum(mask2, dim=1).to(device)
        diff[1,:] = diff[1,:] + diff[0,-1]
        diff[2,:] = diff[2,:] + diff[1,-1]
        diff[faces_non_iterate_2[:,0],faces_non_iterate_2[:,1]] = -1
        idx10, idx11 = torch.where(diff!=-1)
        idxs15 = torch.cat((diff[mask2].unsqueeze(1), idx10.unsqueeze(1), idx11.unsqueeze(1)),dim=1)

        # Find adj_faces that are are a reg for another adj_face in another region
        regions_last = regions[:,:,:,-1].transpose(0,2).reshape(num_meshes,6)

        idx3, idx3_3 = torch.where((regions_last.repeat(3,1,1)[mask2] == adj_face_1[mask2].unsqueeze(1)) & (regions_last.repeat(3,1,1)[mask2] != -1))
        idx3_2, idx3_1 = idxs15[idx3,1], idxs15[idx3,2]

        idx4, idx4_3 = torch.where((regions_last.repeat(3,1,1)[mask2] == adj_face_2[mask2].unsqueeze(1))  & (regions_last.repeat(3,1,1)[mask2] != -1))
        idx4_2, idx4_1 = idxs15[idx4,1], idxs15[idx4,2]

        # Solve the problem of these faces, there are 3 possibilities
        if idx3_1.shape[0]>0 or idx4_1.shape[0]>0:
            change=True
        
        change2=True

        while change2:
            change2=False
            id1,id2 = torch.where((idx3_1.view(-1) == idx3_1.view(-1,1)) & ((idx3_3%3).view(-1) == idx3_2.view(-1,1)) & ((idx3_2).view(-1) == (idx3_3%3).view(-1,1)))
            if id1.shape[0]>0:
                adj_face_1,adj_face_2,regions,ids,faces_non_iterate, idx3_1,idx3_2,idx3_3 ,idx3_1,idx3_2,idx3_3 = find_new_adj_faces_when_2_regions_last_neigh(adjacency_matrix, indices, adj_face_1,adj_face_2,regions,ids,faces_non_iterate, idx3_1,idx3_2,idx3_3%3, idx3_1,idx3_2,idx3_3%3,id1,id2,0,0,0, device)
                id1,id2 = torch.Tensor(),torch.Tensor()
                change2=True
            id1,id2 = torch.where((idx4_1.view(-1) == idx4_1.view(-1,1)) & ((idx4_3%3).view(-1) == idx4_2.view(-1,1)) & ((idx4_2).view(-1) == (idx4_3%3).view(-1,1)))
            if id1.shape[0]>0:
                adj_face_1,adj_face_2,regions,ids,faces_non_iterate, idx4_1,idx4_2,idx4_3 ,idx4_1,idx4_2,idx4_3 = find_new_adj_faces_when_2_regions_last_neigh(adjacency_matrix, indices, adj_face_1,adj_face_2,regions,ids,faces_non_iterate, idx4_1,idx4_2,idx4_3%3, idx4_1,idx4_2,idx4_3%3,id1,id2,1,1,1, device)
                id1,id2 = torch.Tensor(),torch.Tensor()
                change2=True
            id1,id2 = torch.where((idx4_1.view(-1) == idx3_1.view(-1,1)) & ((idx4_3%3).view(-1) == idx3_2.view(-1,1)) & ((idx4_2).view(-1) == (idx3_3%3).view(-1,1)))
            if id1.shape[0]>0:
                adj_face_1,adj_face_2,regions,ids,faces_non_iterate, idx3_1,idx3_2,idx3_3 ,idx4_1,idx4_2,idx4_3 = find_new_adj_faces_when_2_regions_last_neigh(adjacency_matrix, indices, adj_face_1,adj_face_2,regions,ids,faces_non_iterate, idx3_1,idx3_2,idx3_3%3, idx4_1,idx4_2,idx4_3%3,id1,id2,1,0,2, device)
                change2=True

            regions_last = regions[:,:,:,-1].transpose(0,2).reshape(num_meshes,6)
            
            mask2 = torch.ones(adj_face_1.numel(),dtype=bool).view(3,-1)
            mask2[faces_non_iterate[:,0],faces_non_iterate[:,1]] = False
            diff = torch.cumsum(mask2, dim=1).to(device)
            diff[1,:] = diff[1,:] + diff[0,-1]
            diff[2,:] = diff[2,:] + diff[1,-1]
            diff[faces_non_iterate[:,0],faces_non_iterate[:,1]] = -1
            idx10, idx11 = torch.where(diff!=-1)
            idxs15 = torch.cat((diff[mask2].unsqueeze(1), idx10.unsqueeze(1), idx11.unsqueeze(1)),dim=1)

            idx5, idx5_3 = torch.where(regions_last.repeat(3,1,1)[mask2] == adj_face_1[mask2].unsqueeze(1))
            idx5_2, idx5_1 = idxs15[idx5,1], idxs15[idx5,2]

            idx6, idx6_3 = torch.where(regions_last.repeat(3,1,1)[mask2] == adj_face_2[mask2].unsqueeze(1))
            idx6_2, idx6_1 = idxs15[idx6,1], idxs15[idx6,2]

            idx3_1, idx3_2, idx3_3 = torch.cat((idx3_1,idx5_1)), torch.cat((idx3_2,idx5_2)), torch.cat((idx3_3,idx5_3))
            idx4_1, idx4_2, idx4_3 = torch.cat((idx4_1,idx6_1)), torch.cat((idx4_2,idx6_2)), torch.cat((idx4_3,idx6_3))

            idx3_3 = idx3_3 % 3
            idxs_2 = torch.cat((idx3_2.unsqueeze(1),idx3_1.unsqueeze(1),idx3_3.unsqueeze(1)), dim=1).unique(dim=0)
            idx4_3 = idx4_3 % 3
            idxs_3 = torch.cat((idx4_2.unsqueeze(1),idx4_1.unsqueeze(1),idx4_3.unsqueeze(1)), dim=1).unique(dim=0)

            unique, counts = idxs_2[:,:2].unique(dim=0,return_counts=True)
            _,idx12 = torch.where((idxs_2[:,0] == unique[counts>1][:,0].unsqueeze(1)) & (idxs_2[:,1] == unique[counts>1][:,1].unsqueeze(1)))
            unique, counts = idxs_3[:,:2].unique(dim=0,return_counts=True)
            _,idx13 = torch.where((idxs_3[:,0] == unique[counts>1][:,0].unsqueeze(1)) & (idxs_3[:,1] == unique[counts>1][:,1].unsqueeze(1)))

            # if regions.size(2)>10270:#5155
            #     print(idxs_3.size())
            idx210,idx211 = (torch.where((idxs_2[idx12,1] == idxs_3[idx13,1].unsqueeze(1)) & (idxs_2[idx12,0] != idxs_3[idx13,0].unsqueeze(1))))
            if idx210.size(0)>0:
                unique,counts = torch.cat((idxs_3[idx13[idx210.unique()],1:],idxs_2[idx12[idx211.unique()],1:])).unique(dim=0,return_counts=True)
                remove = unique[counts == 1]
                _,idx_remove = torch.where((idxs_3[idx13[idx210.unique()],1] == remove[:,0].unsqueeze(1)) & (idxs_3[idx13[idx210.unique()],2] == remove[:,1].unsqueeze(1)))
                mask = torch.ones(idxs_3.size(0),dtype=bool)
                mask[idx13[idx210.unique()][idx_remove]] = False
                idxs_3 = idxs_3[mask]
                _,idx_remove = torch.where((idxs_2[idx12[idx211.unique()],1] == remove[:,0].unsqueeze(1)) & (idxs_2[idx12[idx211.unique()],2] == remove[:,1].unsqueeze(1)))
                mask = torch.ones(idxs_2.size(0),dtype=bool)
                mask[idx12[idx211.unique()][idx_remove]] = False
                idxs_2 = idxs_2[mask]

                unique, counts = idxs_2[:,:2].unique(dim=0,return_counts=True)
                _,idx12 = torch.where((idxs_2[:,0] == unique[counts>1][:,0].unsqueeze(1)) & (idxs_2[:,1] == unique[counts>1][:,1].unsqueeze(1)))
                unique, counts = idxs_3[:,:2].unique(dim=0,return_counts=True)
                _,idx13 = torch.where((idxs_3[:,0] == unique[counts>1][:,0].unsqueeze(1)) & (idxs_3[:,1] == unique[counts>1][:,1].unsqueeze(1)))
            
            mask = torch.ones(idxs_2.shape[0], dtype=bool)
            mask[idx12[::2]] = False
            idxs_2 = idxs_2[mask]
            idx3_2,idx3_1,idx3_3 = idxs_2[:,0], idxs_2[:,1], idxs_2[:,2]
            
            mask = torch.ones(idxs_3.shape[0], dtype=bool)
            mask[idx13[::2]] = False
            idxs_3 = idxs_3[mask]
            idx4_2,idx4_1,idx4_3 = idxs_3[:,0], idxs_3[:,1], idxs_3[:,2]

        regions_last = regions[:,:,:,-1].transpose(0,2).reshape(num_meshes,6)

        idx58, idx59 = torch.where((idx3_3 == idx4_2.unsqueeze(1)) & (idx3_1 == idx4_1.unsqueeze(1)))

        adj_face_1[idx4_2[idx58], idx4_1[idx58]] = adj_face_1[idx4_3[idx58], idx4_1[idx58]]
        adj_face_2[idx4_2[idx58], idx4_1[idx58]] = adj_face_2[idx4_3[idx58], idx4_1[idx58]]
        regions[idx4_2[idx58],0,idx4_1[idx58],-1] = regions[idx4_3[idx58],0,idx4_1[idx58],-1]
        regions[idx4_2[idx58],1,idx4_1[idx58],-1] = regions[idx4_3[idx58],1,idx4_1[idx58],-1]
        
        adj_face_1[idx3_2, idx3_1] = adj_face_1[idx3_3, idx3_1]
        adj_face_2[idx3_2, idx3_1] = adj_face_2[idx3_3, idx3_1]
        regions[idx3_2,0,idx3_1,-1] = regions[idx3_3,0,idx3_1,-1]
        regions[idx3_2,1,idx3_1,-1] = regions[idx3_3,1,idx3_1,-1]

        adj_face_1[idx4_2, idx4_1] = adj_face_1[idx4_3, idx4_1]
        adj_face_2[idx4_2, idx4_1] = adj_face_2[idx4_3, idx4_1]
        regions[idx4_2,0,idx4_1,-1] = regions[idx4_3,0,idx4_1,-1]
        regions[idx4_2,1,idx4_1,-1] = regions[idx4_3,1,idx4_1,-1]

        faces_non_iterate = find_and_solve_holes(adj_face_1.T, adj_face_2.T, regions, 0, faces_non_iterate)
        faces_non_iterate = (check_zone_is_not_manifold(adjacency_matrix, indices, adj_face_1, adj_face_2, regions, ids, faces_non_iterate, device))

        # Find and solve the case where the adj_faces are in ids (area to remove around the regions)
        idx2, = torch.where(torch.isin(ids[:,1], adj_face_1.reshape(3*adj_face_1.shape[1])))
        mesh_ids = ids[idx2,0]
        idx1_1, idx1_2 = torch.where(adj_face_1[:, mesh_ids] == ids[idx2,1])
        adj_faces_equal_ids = (torch.cat((idx1_1.unsqueeze(1), mesh_ids[idx1_2].unsqueeze(1)), dim = 1).unique(dim=0))
        _, id_remove = (torch.where((adj_faces_equal_ids[:,0] == faces_non_iterate[:,0].unsqueeze(1)) & (adj_faces_equal_ids[:,1] == faces_non_iterate[:,1].unsqueeze(1))))
        mask = torch.ones(adj_faces_equal_ids.numel(), dtype=bool).view(-1,2)
        mask[id_remove] = False
        adj_faces_equal_ids_1 = (adj_faces_equal_ids[mask].view(-1,2))

        idx2, = torch.where(torch.isin(ids[:,1], adj_face_2.reshape(3*adj_face_2.shape[1])))
        mesh_ids = ids[idx2,0]
        idx1_1, idx1_2 = torch.where(adj_face_2[:, mesh_ids] == ids[idx2,1])
        adj_faces_equal_ids = (torch.cat((idx1_1.unsqueeze(1), mesh_ids[idx1_2].unsqueeze(1)), dim = 1).unique(dim=0))
        _, id_remove = (torch.where((adj_faces_equal_ids[:,0] == faces_non_iterate[:,0].unsqueeze(1)) & (adj_faces_equal_ids[:,1] == faces_non_iterate[:,1].unsqueeze(1))))
        mask = torch.ones(adj_faces_equal_ids.numel(), dtype=bool).view(-1,2)
        mask[id_remove] = False
        adj_faces_equal_ids_2 = (adj_faces_equal_ids[mask].view(-1,2))

        idx_remove2, idx_remove1 = (torch.where((adj_faces_equal_ids_1[:,0] == adj_faces_equal_ids_2[:,0].unsqueeze(1)) & (adj_faces_equal_ids_1[:,1] == adj_faces_equal_ids_2[:,1].unsqueeze(1))))
        mask1 = torch.ones(adj_faces_equal_ids_1.numel(), dtype=bool).view(-1,2)
        mask2 = torch.ones(adj_faces_equal_ids_2.numel(), dtype=bool).view(-1,2)
        mask1[idx_remove1] = False
        mask2[idx_remove2] = False
        adj_faces_equal_ids_1 = (adj_faces_equal_ids_1[mask1].view(-1,2))
        adj_faces_equal_ids_2 = (adj_faces_equal_ids_2[mask2].view(-1,2))

        idx6_1 = torch.cat((adj_faces_equal_ids_1[:,1], adj_faces_equal_ids_2[:,1]))
        idx6_2 = torch.cat((adj_faces_equal_ids_1[:,0], adj_faces_equal_ids_2[:,0]))
        new_reg = (torch.cat((adj_face_2[adj_faces_equal_ids_1[:,0], adj_faces_equal_ids_1[:,1]], adj_face_1[adj_faces_equal_ids_2[:,0],adj_faces_equal_ids_2[:,1]])))
        ids = torch.cat((ids, torch.cat((idx6_1.unsqueeze(1), new_reg.unsqueeze(1)),1)),0).type(torch.int).unique(dim=0)
        diff_search = (torch.cat((regions[adj_faces_equal_ids_1[:,0], 1, adj_faces_equal_ids_1[:,1], -1], regions[adj_faces_equal_ids_2[:,0], 0, adj_faces_equal_ids_2[:,1], -1])))
        idx1_1,idx1_2 = (torch.where(adjacency_matrix[indices[idx6_1], new_reg] != diff_search.unsqueeze(1)))
        new_adj_face_1 = adjacency_matrix[indices[idx6_1], new_reg][idx1_1, idx1_2].view(-1,2)[:,0]
        new_adj_face_2 = adjacency_matrix[indices[idx6_1], new_reg][idx1_1, idx1_2].view(-1,2)[:,1]
        
        new_adj_face_1, reg1 = verif_new_faces_are_not_regions(adjacency_matrix, indices, adj_face_1, new_adj_face_1, new_reg.clone().type(torch.long), regions, idx6_1, idx6_2, torch.arange(idx6_1.shape[0]).to(device), 1, torch.Tensor(), ids)
        new_adj_face_2, reg2 = verif_new_faces_are_not_regions(adjacency_matrix, indices, adj_face_2, new_adj_face_2, new_reg.clone().type(torch.long), regions, idx6_1, idx6_2, torch.arange(idx6_1.shape[0]).to(device), 1, torch.Tensor(), ids)
        
        adj_face_1[idx6_2, idx6_1] = new_adj_face_1
        adj_face_2[idx6_2, idx6_1] = new_adj_face_2

        regions[idx6_2, 0, idx6_1, -1] = reg1
        regions[idx6_2, 1, idx6_1, -1] = reg2

        faces_non_iterate = find_and_solve_holes(adj_face_1.T, adj_face_2.T, regions, 0, faces_non_iterate)

        if idx6_1.shape[0]>0:
            change=True

        # Find regions where the adj_faces are equal and iterate we find 2 distinct adj_faces
        adj_face_1, adj_face_2, regions, ids, faces_non_iterate, change_2 = find_and_resolve_meshes_with_same_adj_faces(adjacency_matrix, indices, adj_face_1, adj_face_2, regions, ids, faces_non_iterate, device)
        faces_non_iterate = find_and_solve_holes(adj_face_1.T, adj_face_2.T, regions, 0, faces_non_iterate)
        if change_2 : change = True

    return adj_face_1, adj_face_2, regions, ids, faces_non_iterate

def find_and_resolve_meshes_with_same_adj_faces(adjacency_matrix, indices, adj_face_1, adj_face_2, regions, ids, faces_non_iterate, device):
    """
    Iterates on the regions of the meshes that have equal adj_faces until it finds distinct adj_faces for this region 
    or the region is marked as unsolvable in faces_non_iterate.
    We take the two neighbors of the neighbor that is unique to the second face??
    """
    mask2 = torch.ones(adj_face_1.numel(),dtype=bool).view(3,-1)
    mask2[faces_non_iterate[:,0],faces_non_iterate[:,1]] = False
    diff = torch.cumsum(mask2, dim=1).to(device)
    diff[1,:] = diff[1,:] + diff[0,-1]
    diff[2,:] = diff[2,:] + diff[1,-1]
    diff[faces_non_iterate[:,0],faces_non_iterate[:,1]] = -1
    idx10, idx11 = torch.where(diff!=-1)
    idxs15 = torch.cat((diff[mask2].unsqueeze(1), idx10.unsqueeze(1), idx11.unsqueeze(1)),dim=1)

    idx6, = torch.where(adj_face_2[mask2] == adj_face_1[mask2])
    idx6_2, idx6_1 = idxs15[idx6,1], idxs15[idx6,2]

    change = False
    while idx6_1.shape[0]>0:
        change=True
        new_adj_face_1_same, new_adj_face_2_same, reg1, reg2, ids, faces_non_iterate = find_new_adj_faces_when_faces_same(adjacency_matrix, indices, adj_face_1, adj_face_2, regions[idx6_2, 0, idx6_1, -1], regions[idx6_2, 1, idx6_1, -1], regions, idx6_1, idx6_2, torch.arange(idx6_1.shape[0]).to(device), ids, faces_non_iterate)
        _,idx_to_remove = torch.where((idx6_2 == faces_non_iterate[:,0].view(-1,1)) & (idx6_1 == faces_non_iterate[:,1].view(-1,1)))
        mask = torch.ones(idx6_1.numel(), dtype=bool)
        ids = torch.cat((ids, torch.cat((idx6_1[idx_to_remove].unsqueeze(1), adj_face_1[idx6_2, idx6_1][idx_to_remove].unsqueeze(1)), dim=1)))
        mask[idx_to_remove] = False
        idx6_1, idx6_2, reg1, reg2 = idx6_1[mask], idx6_2[mask], reg1[mask], reg2[mask]
        adj_face_1[idx6_2, idx6_1] = new_adj_face_1_same
        adj_face_2[idx6_2, idx6_1] = new_adj_face_2_same
        regions[idx6_2, 0, idx6_1, -1] = reg1
        regions[idx6_2, 1, idx6_1, -1] = reg2
        faces_non_iterate = find_and_solve_holes(adj_face_1.T, adj_face_2.T, regions, 0, faces_non_iterate)

        mask2 = torch.ones(adj_face_1.numel(),dtype=bool).view(3,-1)
        mask2[faces_non_iterate[:,0],faces_non_iterate[:,1]] = False
        diff = torch.cumsum(mask2, dim=1).to(device)
        diff[1,:] = diff[1,:] + diff[0,-1]
        diff[2,:] = diff[2,:] + diff[1,-1]
        diff[faces_non_iterate[:,0],faces_non_iterate[:,1]] = -1
        idx10, idx11 = torch.where(diff!=-1)
        idxs15 = torch.cat((diff[mask2].unsqueeze(1), idx10.unsqueeze(1), idx11.unsqueeze(1)),dim=1)

        idx6, = torch.where(adj_face_2[mask2] == adj_face_1[mask2])
        idx6_2, idx6_1 = idxs15[idx6,1], idxs15[idx6,2]

    return adj_face_1, adj_face_2, regions, ids, faces_non_iterate,change

def find_new_adj_faces_when_2_regions_last_neigh(adjacency_matrix, indices, adj_face_1,adj_face_2,regions,ids,faces_non_iterate,idx3_1,idx3_2,idx3_3,idx4_1,idx4_2,idx4_3,id1,id2, id_adj_side_1,id_adj_side_2, choice, device):
    """
    Updates the adj_faces that are regs in another regions of the mesh
    It has 3 different options for the update determined in the calling function
    """
    mask_1 = torch.ones(idx3_1.numel(),dtype=bool)
    mask_1[id1]=False
    mask_2 = torch.ones(idx4_1.numel(),dtype=bool)
    mask_2[id2]=False
    if choice == 0:
        adj_face_1[idx3_2[id1],idx3_1[id1]] = adj_face_2[idx3_3[id1],idx3_1[id1]]
        adj_face_1[idx3_3[id1],idx3_1[id1]] = adj_face_2[idx3_2[id1],idx3_1[id1]]
        regions[idx3_3[id1],0,idx3_1[id1],-1] = regions[idx3_2[id1],1,idx3_1[id1],-1]
        regions[idx3_2[id1],0,idx3_1[id1],-1] = regions[idx3_3[id1],1,idx3_1[id1],-1]
    elif choice == 1:
        adj_face_2[idx3_2[id1],idx3_1[id1]] = adj_face_1[idx3_3[id1],idx3_1[id1]]
        adj_face_2[idx3_3[id1],idx3_1[id1]] = adj_face_1[idx3_2[id1],idx3_1[id1]]
        regions[idx3_3[id1],1,idx3_1[id1],-1] = regions[idx3_2[id1],0,idx3_1[id1],-1]
        regions[idx3_2[id1],1,idx3_1[id1],-1] = regions[idx3_3[id1],0,idx3_1[id1],-1]
    elif choice == 2:
        adj_face_2[idx3_3[id1],idx3_1[id1]] = adj_face_2[idx3_2[id1],idx3_1[id1]]
        adj_face_1[idx3_2[id1],idx3_1[id1]] = adj_face_1[idx3_3[id1],idx3_1[id1]]
        regions[idx3_3[id1],1,idx3_1[id1],-1] = regions[idx3_2[id1],1,idx3_1[id1],-1]
        regions[idx3_2[id1],0,idx3_1[id1],-1] = regions[idx3_3[id1],0,idx3_1[id1],-1]

    faces_non_iterate = find_and_solve_holes(adj_face_1.T, adj_face_2.T, regions, 0, faces_non_iterate)
    faces_non_iterate = check_zone_is_not_manifold(adjacency_matrix, indices, adj_face_1, adj_face_2, regions, ids, faces_non_iterate, device)

    # Once updated we iterate until no regions has the two same adj_faces or they have been marked as faces_non_iterate
    idx6_1, = (torch.where(adj_face_1[idx3_2[id1],idx3_1[id1]] == adj_face_2[idx3_2[id1],idx3_1[id1]]))
    idx6_3, = (torch.where(adj_face_1[idx3_3[id1],idx3_1[id1]] == adj_face_2[idx3_3[id1],idx3_1[id1]]))
    idx6_2 = torch.cat((idx3_2[id1][idx6_1], idx3_3[id1][idx6_3]))
    idx6_1 = torch.cat((idx3_1[id1][idx6_1], idx3_1[id1][idx6_3]))
    idx6 = (torch.cat((idx6_1.unsqueeze(1), idx6_2.unsqueeze(1)), dim=1).unique(dim=0))
    idx6_1 = idx6[:,0]
    idx6_2 = idx6[:,1]
    _,idx_to_remove = torch.where((idx6_2 == faces_non_iterate[:,0].view(-1,1)) & (idx6_1 == faces_non_iterate[:,1].view(-1,1)))
    mask = torch.ones(idx6_1.numel(), dtype=bool)
    mask[idx_to_remove] = False
    idx6_1, idx6_2 = idx6_1[mask], idx6_2[mask]

    while idx6_1.shape[0]>0:

        new_adj_face_1_same, new_adj_face_2_same, reg1, reg2, ids, faces_non_iterate = find_new_adj_faces_when_faces_same(adjacency_matrix, indices, adj_face_1, adj_face_2, regions[idx6_2, 0, idx6_1, -1], regions[idx6_2, 1, idx6_1, -1], regions, idx6_1, idx6_2, torch.arange(idx6_1.shape[0]).to(device), ids, faces_non_iterate)
        _,idx_to_remove = torch.where((idx6_2 == faces_non_iterate[:,0].view(-1,1)) & (idx6_1 == faces_non_iterate[:,1].view(-1,1)))
        mask = torch.ones(idx6_1.numel(), dtype=bool)
        ids = torch.cat((ids, torch.cat((idx6_1[idx_to_remove].unsqueeze(1), adj_face_1[idx6_2, idx6_1][idx_to_remove].unsqueeze(1)), dim=1)))
        mask[idx_to_remove] = False
        idx6_1, idx6_2, reg1, reg2 = idx6_1[mask], idx6_2[mask], reg1[mask], reg2[mask]
        
        adj_face_1[idx6_2, idx6_1] = new_adj_face_1_same
        adj_face_2[idx6_2, idx6_1] = new_adj_face_2_same
        regions[idx6_2, 0, idx6_1, -1] = reg1
        regions[idx6_2, 1, idx6_1, -1] = reg2

        faces_non_iterate = find_and_solve_holes(adj_face_1.T, adj_face_2.T, regions, 0, faces_non_iterate)
        idx6_1, = (torch.where(adj_face_1[idx3_2[id1],idx3_1[id1]] == adj_face_2[idx3_2[id1],idx3_1[id1]]))
        idx6_3, = (torch.where(adj_face_1[idx3_3[id1],idx3_1[id1]] == adj_face_2[idx3_3[id1],idx3_1[id1]]))
        idx6_2 = torch.cat((idx3_2[id1][idx6_1], idx3_3[id1][idx6_3]))
        idx6_1 = torch.cat((idx3_1[id1][idx6_1], idx3_1[id1][idx6_3]))
        idx6 = (torch.cat((idx6_1.unsqueeze(1), idx6_2.unsqueeze(1)), dim=1).unique(dim=0))
        idx6_1 = idx6[:,0]
        idx6_2 = idx6[:,1]

        _,idx_to_remove = torch.where((idx6_2 == faces_non_iterate[:,0].view(-1,1)) & (idx6_1 == faces_non_iterate[:,1].view(-1,1)))
        mask = torch.ones(idx6_1.numel(), dtype=bool)
        mask[idx_to_remove] = False
        idx6_1, idx6_2 = idx6_1[mask], idx6_2[mask]
    
    # Remove indices of meshes already treated
    idx3_1,idx3_2,idx3_3 = idx3_1[mask_1],idx3_2[mask_1],idx3_3[mask_1]
    idx4_1,idx4_2,idx4_3 = idx4_1[mask_2],idx4_2[mask_2],idx4_3[mask_2]
    return adj_face_1,adj_face_2,regions,ids,faces_non_iterate, idx3_1,idx3_2,idx3_3 ,idx4_1,idx4_2,idx4_3

def find_new_faces_adj_regions(adjacency_matrix, indices, adj_face_1, adj_face_2, adj_face_1_old, adj_face_2_old, regions, regions_old, ids, faces_non_iterate, idx1_1, idx1_2, idx1_3, id_adj_side):
    """
    Find new adj_faces when one of the adj_face is actually a region of the pooling
    """
    id_other_side = abs(id_adj_side-1)
    new_adj_faces_1 = adj_face_2_old[idx1_3, idx1_2].clone().type(torch.long)

    idx5 = torch.where((adj_face_1_old[idx1_3, idx1_2] != regions_old[idx1_1, id_adj_side, idx1_2, -1]))# & (adj_face_1_old[idx1_3, idx1_2] != adj_face_2[idx1_1,idx1_2]))
    new_adj_faces_1[idx5] = adj_face_1_old[idx1_3, idx1_2][idx5]
    new_regs_1 = regions_old[idx1_3, id_other_side, idx1_2,-1]
    new_regs_1[idx5] = regions_old[idx1_3[idx5], id_adj_side, idx1_2[idx5],-1]

    ids = torch.cat((ids, torch.cat((idx1_2.view(idx1_2.shape[0],1),new_regs_1.view(idx1_2.shape[0],1)), dim=1)), dim = 0).unique(dim = 0)
    regions[idx1_1,id_adj_side, idx1_2,-1] = new_regs_1
    adj_face_1[idx1_1, idx1_2] = new_adj_faces_1
    
    #Détecte si on arrive sur du vide, dans ce cas on arrête
    idx11_2, = torch.where(adj_face_1[idx1_1,idx1_2] == -1)
    if idx11_2.shape[0] > 0:
        faces_non_iterate = torch.cat((faces_non_iterate, torch.cat((idx1_1[idx11_2].view(-1,1), idx1_2[idx11_2].view(-1,1)), dim=1)),dim=0)
        mask_1 = torch.ones(idx1_2.numel(), dtype=torch.bool)
        mask_1[idx11_2] = False
        idx1_2 = idx1_2[mask_1]
        idx1_1 = idx1_1[mask_1]

    # Ici on vérifie que les faces sélectionnées ne sont pas encore des régions, sinon on prend juste la face suivante
    _, idx10 = torch.where(adj_face_1[idx1_1, idx1_2] == regions[idx1_1, id_adj_side, idx1_2,1:4].T)
    if idx10.shape[0] > 0 :
        idx8_1, idx8 = idx1_1[idx10], idx1_2[idx10]
        idx9_1, idx9_2 = torch.where((adjacency_matrix[indices[idx8], adj_face_1[idx8_1, idx8]] != regions[idx8_1, id_adj_side, idx8, 0].view(idx8.shape[0],1)) & 
                                     (adjacency_matrix[indices[idx8], adj_face_1[idx8_1, idx8]] != adj_face_1_old[idx8_1, idx8].view(idx8.shape[0],1)))
        old_adj_1 = adj_face_1[idx8_1, idx8].clone().type(torch.long)
        
        adj_face_1[idx8_1, idx8] = adjacency_matrix[indices[idx8], adj_face_1[idx8_1, idx8]][idx9_1, idx9_2]
        regions[idx8_1,id_adj_side,idx8,-1] = old_adj_1
    
    #Détecte si on arrive sur du vide, dans ce cas on arrête
    idx11_2, = torch.where(adj_face_1[idx1_1,idx1_2] == -1)
    if idx11_2.shape[0] > 0:
        faces_non_iterate = torch.cat((faces_non_iterate, torch.cat((idx1_1[idx11_2].view(-1,1), idx1_2[idx11_2].view(-1,1)), dim=1)),dim=0)
        mask_1 = torch.ones(idx1_2.numel(), dtype=torch.bool)
        mask_1[idx11_2] = False
        idx1_2 = idx1_2[mask_1]
        idx1_1 = idx1_1[mask_1]

    return adj_face_1, adj_face_2, regions, ids, faces_non_iterate

def check_zone_is_not_manifold(adjacency_matrix, indices, adj_face_1, adj_face_2, regions, ids, faces_non_iterate, device):
    """
    Detects if the pooling region is a whole structure of its own, in this case we do not want to iterate on it
    and we add it to faces_non_iterate.
    """
    num_mesh = adj_face_1.shape[1]
    regs_0 = torch.cat((torch.arange(num_mesh).view(-1,1), torch.arange(num_mesh).view(-1,1), torch.arange(num_mesh).view(-1,1), torch.arange(num_mesh).view(-1,1)), dim=0).type(torch.int).to(device)
    regs_1 = torch.cat((regions[0,0,:,0].view(-1,1), regions[0,0,:,1].view(-1,1), regions[0,0,:,2].view(-1,1), regions[0,0,:,3].view(-1,1)), dim=0).type(torch.int)
    regs = torch.cat((regs_0, regs_1), dim=1).type(torch.int)
    ids = torch.cat((ids, regs), dim=0).type(torch.int).unique(dim=0)
    faces_to_check = torch.cat((adj_face_1, adj_face_2, 
                                adjacency_matrix[indices[torch.arange(num_mesh)], adj_face_1[0,:]].T,
                                adjacency_matrix[indices[torch.arange(num_mesh)], adj_face_1[1,:]].T,
                                adjacency_matrix[indices[torch.arange(num_mesh)], adj_face_1[2,:]].T), dim=0).type(torch.int)
    idx2, = torch.where(torch.isin(ids[:,1], faces_to_check))
    mesh_ids = ids[idx2,0]
    idx1_1, idx1_2 = torch.where(faces_to_check[:, mesh_ids] == ids[idx2,1])
    unique, counts = mesh_ids[idx1_2].unique(return_counts = True)
    meshes_non_iterate = unique[counts==15]

    faces_non_iterate = torch.cat((faces_non_iterate, torch.cat((torch.arange(3).repeat(meshes_non_iterate.shape[0]).view(-1,1).to(device), meshes_non_iterate.view(-1,1).repeat(1,3).view(-1,1)), dim=1)),dim=0)
    return faces_non_iterate

def find_new_adj_faces_when_faces_same(adjacency_matrix, indices, adj_face_1, adj_face_2, reg1, reg2, regions, idx_mesh_wrong_face_same, idx_region_number, idx_2, ids, faces_non_iterate):
    """
    find the new adj_faces when 2 preceeding are the same for a region
    """
    # Find the adjacent face of original faces different from the 2 old regs
    idx1_1, idx1_2 = torch.where((adjacency_matrix[indices[idx_mesh_wrong_face_same], adj_face_1[idx_region_number, idx_mesh_wrong_face_same]] != reg1[idx_2].unsqueeze(1)) &
                                 (adjacency_matrix[indices[idx_mesh_wrong_face_same], adj_face_1[idx_region_number, idx_mesh_wrong_face_same]] != reg2[idx_2].unsqueeze(1)))

    unique,counts = idx1_1.unique(return_counts = True)

    new_reg = adjacency_matrix[indices[idx_mesh_wrong_face_same], adj_face_1[idx_region_number, idx_mesh_wrong_face_same]][idx1_1, idx1_2]
    diff_search = adj_face_1[idx_region_number, idx_mesh_wrong_face_same]
    # We stop iterating on the regions where we return to the center face (it means we are in a loop)
    idx_to_remove_1, = torch.where(new_reg == regions[0,0,idx_mesh_wrong_face_same,0])
    idx_to_remove_2, = torch.where(new_reg == -1)
    idx_to_remove = torch.cat((idx_to_remove_1, idx_to_remove_2))
    faces_non_iterate = torch.cat((faces_non_iterate, torch.cat((idx_region_number[idx_to_remove].view(-1,1), idx_mesh_wrong_face_same[idx_to_remove].view(-1,1)), dim=1)),dim=0)
    mask = torch.ones(new_reg.numel(), dtype=bool)
    mask[idx_to_remove] = False

    idx_mesh_wrong_face_same, idx_region_number, new_reg, idx_2, diff_search = idx_mesh_wrong_face_same[mask], idx_region_number[mask], new_reg[mask], idx_2[mask], diff_search[mask]
    
    if idx_mesh_wrong_face_same.shape[0] > 0:
        # print("here")
        new_reg, diff_search = verif_new_faces_are_not_regions(adjacency_matrix, indices, adj_face_1, new_reg, reg1, regions, idx_mesh_wrong_face_same, idx_region_number, idx_2, 0, diff_search, ids)

    # We stop iterating on the regions where new_reg is -1 (we are in a hole so we cannot go further)
    idx_to_remove, = torch.where(new_reg == -1)
    faces_non_iterate = torch.cat((faces_non_iterate, torch.cat((idx_region_number[idx_to_remove].view(-1,1), idx_mesh_wrong_face_same[idx_to_remove].view(-1,1)), dim=1)),dim=0)
    mask = torch.ones(new_reg.numel(), dtype=bool)
    mask[idx_to_remove] = False

    idx_mesh_wrong_face_same, idx_region_number, new_reg, idx_2, diff_search = idx_mesh_wrong_face_same[mask], idx_region_number[mask], new_reg[mask], idx_2[mask], diff_search[mask]

    reg1[idx_2] = new_reg
    reg2[idx_2] = new_reg
    # Il faudrait traiter le cas où new_reg == -1
    ids = torch.cat((ids, torch.cat((idx_mesh_wrong_face_same.unsqueeze(1), new_reg.unsqueeze(1)),1)),0).type(torch.int)
    ids = torch.cat((ids, torch.cat((idx_mesh_wrong_face_same.unsqueeze(1), adj_face_1[idx_region_number, idx_mesh_wrong_face_same].unsqueeze(1)),1)),0).type(torch.int)
    
    # We find the 2 adgacent faces of the new reg that is different to the old adj_faces
    idx1_1, idx1_2 = torch.where(adjacency_matrix[indices[idx_mesh_wrong_face_same], reg1[idx_2]] != diff_search.unsqueeze(1))

    new_adj_face_1_same = adjacency_matrix[indices[idx_mesh_wrong_face_same], reg1[idx_2]][idx1_1, idx1_2].view(idx_mesh_wrong_face_same.shape[0],2)[:,0]
    new_adj_face_2_same = adjacency_matrix[indices[idx_mesh_wrong_face_same], reg1[idx_2]][idx1_1, idx1_2].view(idx_mesh_wrong_face_same.shape[0],2)[:,1]
    idx_still_same, = (torch.where(new_adj_face_1_same == new_adj_face_2_same))
    idx_control,_,idx_remove = torch.where(new_adj_face_1_same[idx_still_same] == regions[0,0,idx_mesh_wrong_face_same[idx_still_same],0:4].unsqueeze(2))
    # Again we remove the regions where the 2 new adj_faces are still equal and they are a region (we are in a loop where everything is removed)
    idx_remove = idx_remove[idx_control == idx_remove]
    faces_non_iterate = torch.cat((faces_non_iterate, torch.cat((idx_region_number[idx_still_same[idx_remove]].view(-1,1), idx_mesh_wrong_face_same[idx_still_same[idx_remove]].view(-1,1)), dim=1)),dim=0)
    mask = torch.ones(new_adj_face_1_same.numel(), dtype=bool)
    mask[idx_still_same[idx_remove]] = False
    idx_mesh_wrong_face_same, idx_region_number, idx_2, diff_search = idx_mesh_wrong_face_same[mask], idx_region_number[mask], idx_2[mask], diff_search[mask]
    new_adj_face_1_same, new_adj_face_2_same = new_adj_face_1_same[mask], new_adj_face_2_same[mask]

    # We check that the new adj_faces are not regions anymore
    new_adj_face_1_same, reg1 = verif_new_faces_are_not_regions(adjacency_matrix, indices, adj_face_1, new_adj_face_1_same, reg1, regions, idx_mesh_wrong_face_same, idx_region_number, idx_2, 1, torch.Tensor(), ids)
    new_adj_face_2_same, reg2 = verif_new_faces_are_not_regions(adjacency_matrix, indices, adj_face_2, new_adj_face_2_same, reg2, regions, idx_mesh_wrong_face_same, idx_region_number, idx_2, 1, torch.Tensor(), ids)

    ids = torch.cat((ids, torch.cat((idx_mesh_wrong_face_same.unsqueeze(1), reg1[idx_2].unsqueeze(1)),1)),0).type(torch.int)
    ids = torch.cat((ids, torch.cat((idx_mesh_wrong_face_same.unsqueeze(1), reg2[idx_2].unsqueeze(1)),1)),0).type(torch.int).unique(dim=0)

    return new_adj_face_1_same, new_adj_face_2_same, reg1, reg2, ids, faces_non_iterate

def verif_new_faces_are_not_regions(adjacency_matrix, indices, adj_face_1, new_adj_faces, reg1, regions, idx_mesh_wrong_face, idx_region_number, idx, choice, diff_search, ids):    
    """
    Here we check that the newly selected faces are not a region again
    """
    idx3, idx3_2 = torch.where(new_adj_faces.view(new_adj_faces.shape[0],1) == regions[idx_region_number,0,idx_mesh_wrong_face,1:4])#.view(idx_mesh_wrong_face.shape[0],3))
    idx_keep, = torch.where(new_adj_faces[idx3] != -1)
    idx3, idx3_2 = idx3[idx_keep], idx3_2[idx_keep]
    # rajouter les -1 -> oui et comparer avec les ids pas si on ne trouve pas dans les régions
    while idx3.shape[0] > 0:
        idx_check = idx_mesh_wrong_face[idx3]

        if type(idx_region_number) != int:
            idx_clone = idx_region_number.clone().type(torch.int)
            idx_region_number = idx_region_number[idx3]
            
        if choice == 1:
            temp_adj_face = reg1[idx[idx3]].clone().type(torch.int)
        elif choice == 0:
            temp_adj_face = adj_face_1[idx_region_number, idx_check]
        if reg1.shape[0]>0:
            reg1[idx[idx3]] = regions[idx_region_number,0,idx_check,idx3_2+1]
        if diff_search.shape[0] > 0:
            temp_adj_face = diff_search[idx3]
            diff_search[idx3] = regions[idx_region_number,0,idx_check,idx3_2+1]
        #rajouter les regions[-1]? -> va vraiment falloir le faire à un moment
        idx_temp1,idx_temp2,idx_temp = (torch.where((adjacency_matrix[indices[idx_check], new_adj_faces[idx3]].unsqueeze(1) == regions[0,0,idx_check,0:4].unsqueeze(2))))
        unique,counts = idx_temp1.unique(return_counts=True)
        if unique[counts==2].shape[0]>0:
            mesh_wrong = unique[counts==2]
            _,idx4_1 = torch.where(idx_temp1==mesh_wrong.unsqueeze(1))
            idx4_1 = idx4_1.view(-1,2)
            mask = torch.ones(idx_temp.numel(),dtype=bool)
            mask[idx4_1[:,0]] = False
            # Refind the correct new reg
            idx_check_1,idx_check_2,idx_temp3 = (torch.where((adjacency_matrix[indices[idx_check[mesh_wrong]], new_adj_faces[idx3[mesh_wrong]]].unsqueeze(1) == regions[0,0,idx_check[mesh_wrong],0].unsqueeze(1))))
            idx_temp3 = (idx_temp3[idx_check_1 == idx_check_2])
            idx_temp[idx4_1[:,1]] = idx_temp3
            idx_temp = idx_temp[mask]
        idx1_1, idx1_2 = torch.where(adjacency_matrix[indices[idx_check], new_adj_faces[idx3]] != temp_adj_face.unsqueeze(1))
        idx2_1, _, idx2_2 = torch.where(idx1_2.view(-1,1,2) == idx_temp.view(idx_check.shape[0],-1,1))

        mask = torch.ones(idx1_1.numel(), dtype=bool).view(-1,2)
        mask[idx2_1, idx2_2] = False
        idx1_1, idx1_2 = idx1_1.view(-1,2)[mask], idx1_2.view(-1,2)[mask]

        new_adj_faces[idx3] = adjacency_matrix[indices[idx_check], new_adj_faces[idx3]][idx1_1, idx1_2]

        if type(idx_region_number) != int:
            idx_region_number = idx_clone

        idx3, idx3_2 = torch.where(new_adj_faces.view(new_adj_faces.shape[0],1) == regions[idx_region_number,0,idx_mesh_wrong_face,1:4])

        idx10, = torch.where(new_adj_faces[idx3] == -1)
        mask = torch.ones(idx3.shape[0], dtype = bool)
        mask[idx10] = False
        idx3, idx3_2 = idx3[mask], idx3_2[mask]

    if diff_search.shape[0] > 0:
        return new_adj_faces, diff_search

    return new_adj_faces, reg1   

def update_adj_matrix(adjacency_matrix, regions, adj_face_1, adj_face_2, ids, num_meshes, device):
    """
    Update of the adjacency matrix for the adj_faces
    Also remove all regions and faces in ids
    """
    for i in range(3):
        idx1_1, idx1_2 = torch.where(adjacency_matrix[torch.arange(num_meshes), adj_face_1[i]] == regions[i, 0, :,-1].unsqueeze(1))
        idx2_1, idx2_2 = torch.where(adjacency_matrix[torch.arange(num_meshes), adj_face_2[i]] == regions[i, 1, :,-1].unsqueeze(1))

        adjacency_matrix[idx1_1, adj_face_1[i][idx1_1], idx1_2] = adj_face_2[i, idx1_1]
        adjacency_matrix[idx2_1, adj_face_2[i][idx2_1], idx2_2] = adj_face_1[i, idx2_1]

        idx3, = torch.where((adjacency_matrix[idx1_1, adj_face_1[i][idx1_1], 0] == -1) & (adjacency_matrix[idx1_1, adj_face_1[i][idx1_1], 1] == -1) & (adjacency_matrix[idx1_1, adj_face_1[i][idx1_1], 2] == -1))         
        idx4, = torch.where((adjacency_matrix[idx2_1, adj_face_2[i][idx2_1], 0] == -1) & (adjacency_matrix[idx2_1, adj_face_2[i][idx2_1], 1] == -1) & (adjacency_matrix[idx2_1, adj_face_2[i][idx2_1], 2] == -1))

        ids = torch.cat((ids, torch.cat((idx1_1[idx3].unsqueeze(1), adj_face_1[i][idx1_1][idx3].unsqueeze(1)), dim=1)), dim=0)
        ids = torch.cat((ids, torch.cat((idx2_1[idx4].unsqueeze(1), adj_face_2[i][idx2_1][idx4].unsqueeze(1)), dim=1)), dim=0)

    # Put (-1, -1, -1) for all the faces of the meshes that have been removed during the pooling
    ids = ids.type(torch.int64)
    adjacency_matrix[ids[:,0], ids[:,1]] = (torch.ones(ids.shape[0], 3) * -1).type(torch.long).to(device)

    adjacency_matrix[torch.arange(num_meshes)[:,None], regions[0,0,:,:4], :] = -1
    adjacency_matrix[torch.arange(num_meshes)[:,None], regions[0,1,:,:4], :] = -1

    num_faces = adjacency_matrix.shape[1]

    return adjacency_matrix, ids, num_faces

def update_adj_matrix_parallel(adjacency_matrix, indices, regions, adj_face_1, adj_face_2, erased_iter, num_meshes, device):
    """
    Update of the adjacency matrix for the adj_faces
    Also remove all regions and faces in ids
    """
    # print(adjacency_matrix[2,2418])
    idx0_0, idx1_0, idx2_0 = torch.where(adjacency_matrix[indices, adj_face_1] == regions[:, 0, :,-1].unsqueeze(2))
    idx0_1, idx1_1, idx2_1 = torch.where(adjacency_matrix[indices, adj_face_2] == regions[:, 1, :,-1].unsqueeze(2))
    
    adjacency_matrix[indices[idx1_0], adj_face_1[idx0_0,idx1_0], idx2_0] = adj_face_2[idx0_0,idx1_0]
    inverse_indices_1 = torch.cat((idx0_0.unsqueeze(1),idx1_0.unsqueeze(1),idx2_0.unsqueeze(1)),dim=1)
    
    adjacency_matrix[indices[idx1_1], adj_face_2[idx0_1,idx1_1], idx2_1] = adj_face_1[idx0_1,idx1_1]
    inverse_indices_2 = torch.cat((idx0_1.unsqueeze(1),idx1_1.unsqueeze(1),idx2_1.unsqueeze(1)),dim=1)

    idx_missing_1 = torch.arange(3*adj_face_1.size(1)).to(device)[torch.isin(torch.arange(3*adj_face_1.size(1)).to(device),inverse_indices_1[:,1] + adj_face_1.size(1)*inverse_indices_1[:,0], invert=True)]
    inverse_indices_1 = torch.cat((inverse_indices_1, torch.cat(((idx_missing_1 // adj_face_1.size(1)).unsqueeze(1), (idx_missing_1 % adj_face_1.size(1)).unsqueeze(1), torch.zeros(idx_missing_1.size(0)).to(device).unsqueeze(1).type(torch.int)), dim=1))).unique(dim=0)

    idx_missing_2 = torch.arange(3*adj_face_1.size(1)).to(device)[torch.isin(torch.arange(3*adj_face_1.size(1)).to(device),inverse_indices_2[:,1] + adj_face_1.size(1)*inverse_indices_2[:,0], invert=True)]
    inverse_indices_2 = torch.cat((inverse_indices_2, torch.cat(((idx_missing_2 // adj_face_1.size(1)).unsqueeze(1), (idx_missing_2 % adj_face_1.size(1)).unsqueeze(1), torch.zeros(idx_missing_2.size(0)).to(device).unsqueeze(1).type(torch.int)), dim=1))).unique(dim=0)

    adj_face_1 = adj_face_1.view(-1)
    adj_face_2 = adj_face_2.view(-1)

    return adjacency_matrix, erased_iter, inverse_indices_1, inverse_indices_2

def update_features(features, regions, adj_face_1, adj_face_2, num_meshes, num_channels, ids, device):
    # aggregate features according to connectivity of the faces
    features = update_face_features(features, regions, num_meshes, num_channels, 1, adj_face_1, adj_face_2, ids, device)

    # We remove the features of the faces that have been removed by the face collapse
    i = ids.type(torch.int)
    features[i[:,0], i[:,1]] = (torch.zeros(i.shape[0], num_channels)).type(torch.float).to(device)

    # Remove features of 4 removed faces
    mask = torch.ones(features.numel(), dtype=torch.bool).view(num_meshes, -1, num_channels)
    features[torch.arange(num_meshes)[:,None], regions[0,0,:,:4], :] = 0
    features[torch.arange(num_meshes)[:,None], regions[0,1,:,:4], :] = 0
    temp, counts = torch.where(mask==False)[0].unique(return_counts = True)

    return features

def update_features_parallel(features, indices, regions, adj_face_1, adj_face_2, num_meshes, num_channels, ids, erased_iter, points, device):
    # aggregate features according to connectivity of the faces
    if points:
        features1, features2 = update_face_features_points_new(features, indices, regions, num_meshes, num_channels, 1, adj_face_1, adj_face_2, ids, device, 1)
    else:
        features1, features2 = update_face_features_new(features, indices, regions, num_meshes, num_channels, 1, adj_face_1, adj_face_2, ids, device, 1)

    features[indices[:,None], adj_face_1.T] = features1
    features[indices[:,None], adj_face_2.T] = features2

    # Locate faces that are updated multiple times during the pooling
    temp = torch.cat((torch.cat(((torch.ones(3,indices.shape[0]).to(device) * indices).type(torch.int).view(-1,1),adj_face_1.view(-1,1), torch.arange(indices.shape[0]).repeat(3).view(-1,1).to(device), 
                                 torch.arange(3).unsqueeze(1).repeat(1,indices.shape[0]).view(-1,1).to(device), torch.zeros(3*indices.shape[0]).unsqueeze(1).to(device)), dim=1), 
                      torch.cat(((torch.ones(3,indices.shape[0]).to(device) * indices).type(torch.int).view(-1,1),adj_face_2.view(-1,1), torch.arange(indices.shape[0]).repeat(3).view(-1,1).to(device), 
                                 torch.arange(3).unsqueeze(1).repeat(1,indices.shape[0]).view(-1,1).to(device), torch.ones(3*indices.shape[0]).unsqueeze(1).to(device)), dim=1))).type(torch.int)
    
    unique1,inverse1,counts1 = temp[:,:2].unique(dim=0,return_counts=True,return_inverse=True)
    idx, = torch.where(counts1>1)

    idx10, = torch.where(torch.isin(inverse1,idx))
    temp2 = temp[idx10]
    unique,inverse,counts = temp2[:,:2].unique(dim=0,return_counts=True,return_inverse=True)
    temp2 = torch.cat((temp2,inverse.unsqueeze(1)),dim=1)
    
    if temp.size(0)>0:

        if points:
            temp0 = temp2[temp2[:,-2] == 0]
            idx2_0 = temp0[:,-1].type(torch.long).unsqueeze(1).unsqueeze(1).repeat(1,features.shape[2],features.shape[3])
            feat_temp2 = torch.zeros(unique.shape[0], features.shape[2],features.shape[3]).to(device).scatter_add(0, idx2_0, features1[temp0[:,2],temp0[:,3]])

            temp1 = temp2[temp2[:,-2] == 1]
            idx2_0 = temp1[:,-1].type(torch.long).unsqueeze(1).unsqueeze(1).repeat(1,features.shape[2],features.shape[3])
            feat_temp2 = feat_temp2.scatter_add(0, idx2_0, features2[temp1[:,2],temp1[:,3]])

            features[unique[:,0],unique[:,1]] = (feat_temp2 / counts.unsqueeze(1).unsqueeze(1).repeat(1,features.shape[2], features.size(3))).type(features.type())
        else:
            temp0 = temp2[temp2[:,-2] == 0]
            idx2_0 = temp0[:,-1].type(torch.long).unsqueeze(1).repeat(1,features.shape[2])
            feat_temp2 = torch.zeros(unique.shape[0], features.shape[2]).to(device).scatter_add(0, idx2_0, features1[temp0[:,2],temp0[:,3]])

            temp1 = temp2[temp2[:,-2] == 1]
            idx2_0 = temp1[:,-1].type(torch.long).unsqueeze(1).repeat(1,features.shape[2])
            feat_temp2 = feat_temp2.scatter_add(0, idx2_0, features2[temp1[:,2],temp1[:,3]])

            features[unique[:,0],unique[:,1]] = feat_temp2 / counts.unsqueeze(1).repeat(1,features.shape[2])

    # We remove the features of the faces that have been removed by the face collapse
    features[erased_iter[:,0],erased_iter[:,1]]=0
    
    return features

def update_face_features_points_new(features, indices, regions, num_meshes, num_channels, _, adj_face_1, adj_face_2, ids, device, choice = 0):

    index_mean_region = ids[:,2].unsqueeze(1).unsqueeze(2).repeat(1,features.shape[2],features.shape[3])
    mean_feats_region = torch.zeros(adj_face_1.shape[1], features.shape[2], features.shape[3]).to(device).scatter_add(0, index_mean_region, features[ids[:,0],ids[:,1]])
    mean_feats_region = mean_feats_region.unsqueeze(1).repeat(1,3,1,1)

    _,counts = ids[:,2].unique(return_counts=True)
    counts += 1
    counts = counts.view(-1,1,1,1).repeat(1,3,features.shape[2],features.shape[3])
    
    ret_features_1 = (mean_feats_region + features[indices[:,None], adj_face_1.T]) / counts
    ret_features_2 = (mean_feats_region + features[indices[:,None], adj_face_2.T]) / counts

    return ret_features_1, ret_features_2

def update_face_features_new(features, indices, regions, num_meshes, num_channels, _, adj_face_1, adj_face_2, ids, device, choice = 0):

    index_mean_region = ids[:,2].unsqueeze(1).repeat(1,features.shape[2])
    mean_feats_region = torch.zeros(adj_face_1.shape[1], features.shape[2]).to(device).scatter_add(0, index_mean_region, features[ids[:,0],ids[:,1]])
    mean_feats_region = mean_feats_region.unsqueeze(1).repeat(1,3,1)

    _,counts = ids[:,2].unique(return_counts=True)
    counts = counts.view(-1,1,1).repeat(1,3,features.shape[2])

    mean_feats_region = mean_feats_region / counts
    ret_features_1 = (mean_feats_region + features[indices[:,None], adj_face_1.T]) / 2
    ret_features_2 = (mean_feats_region + features[indices[:,None], adj_face_2.T]) / 2

    return ret_features_1, ret_features_2


def create_global_regions(adj_face_1, adj_face_2, regions, adjacency_matrix, faces, indices, head_meshes, meshes_to_pool, ids, num_faces, target_size, faces_non_iterate, device):
    list_ids = torch.arange(adj_face_1.shape[1]).type(torch.int).to(device)
    mask_1 = torch.ones(list_ids.size(0),dtype=bool).to(device)
    mask_1[faces_non_iterate[:,1]] = 0
    list_ids = list_ids[mask_1]

    list_regions_retained = torch.Tensor().to(device)
    list_indices_regions_retained = []
    
    cpt2 = 0
    remove_tot = [0] * meshes_to_pool.shape[0]

    idx_glob = torch.Tensor(0,3).type(torch.int).to(device)
    erased_iter = torch.Tensor(0,3).type(torch.int).to(device)

    meshes_continue_pooling = meshes_to_pool.clone().type(torch.int).to(device)
    indices_2 = indices.clone().type(torch.int).to(device)
    indices = indices[mask_1]

    map = (torch.ones(meshes_to_pool.max()+1)*-1).to(device).type(torch.int)
    map[meshes_to_pool] = torch.arange(meshes_to_pool.shape[0]).to(device).type(torch.int)

    erased_total = torch.cat((((torch.ones(30,adj_face_1.shape[1]).to(device) * torch.arange(adj_face_1.shape[1]).to(device)).type(torch.int).transpose(0,1).reshape(30*adj_face_1.shape[1],1)).to(device), regions[:,:,:,:].transpose(0,2).reshape(adj_face_1.shape[1]*30,1)), dim=1).unique(dim=0)
    erased_total = torch.cat((erased_total,ids)).unique(dim=0)
    erased_total = torch.cat((erased_total,erased_total[:,0].clone().type(torch.int).unsqueeze(1)),dim=1)
    
    num_faces_clone = num_faces.clone().to(device).type(torch.int)
    mask = (num_faces_clone) > target_size
    num_faces = (num_faces_clone)[mask]
    head_meshes = head_meshes[mask]
    meshes_continue_pooling = meshes_to_pool[mask]

    not_poolable = torch.isin(meshes_continue_pooling,indices.unique(), invert=True)
    meshes_not_poolable = meshes_continue_pooling[not_poolable]

    meshes_continue_pooling = meshes_continue_pooling[~not_poolable]
    num_faces = (num_faces)[~not_poolable]
    head_meshes = head_meshes[~not_poolable]
    if meshes_continue_pooling.size(0)>0:
        # Commpute the points removed by each face pooling
        faces_to_compress = regions[0,0,:,0]
        faces_to_compress = faces[indices,faces_to_compress[mask_1]]
        replacement = faces_to_compress[:,0]
        faces_to_compress = torch.cat((faces_to_compress.view(-1,1), list_ids.view(-1,1).repeat(1,3).view(-1,1)),dim=1) #torch.arange(indices.size(0)).to(device)

        idx_er_in_reg,_ = torch.where(ids[:,1].unsqueeze(1) == regions[0,0,ids[:,0],:4])
        erased_iter_non_region = ids[torch.isin(torch.arange(ids.size(0)).to(device), idx_er_in_reg, invert=True)]

        idx48, = torch.where(torch.isin(erased_total[:,0],erased_iter_non_region[:,0]))
        adj_erased_faces = torch.cat((erased_iter_non_region[:,0].unsqueeze(1).repeat(1,3).view(-1,1), adjacency_matrix[indices_2[erased_iter_non_region[:,0]],erased_iter_non_region[:,1]].view(-1,1)),dim=1)
        adj_erased_faces,inverse_1 = adj_erased_faces.unique(dim=0, return_inverse=True)
        adj_erased_faces = torch.cat((adj_erased_faces, erased_total[idx48,:2].unique(dim=0)))
        unique,inverse,counts = adj_erased_faces.unique(dim=0,return_inverse=True,return_counts = True)
        idx60, = torch.where(counts>=2)
        idx61, = torch.where(torch.isin(inverse, idx60))
        idx61 = idx61[idx61<inverse_1.size(0)]
        idx62, = torch.where(torch.isin(inverse_1, idx61))
        unique,counts = (idx62//3).unique(return_counts=True)
        idx50 = unique[counts == 3]

        erased_iter_non_region = erased_iter_non_region[idx50]

        to_add = torch.cat((faces[indices_2[erased_iter_non_region[:,0]], erased_iter_non_region[:,1]].view(-1,1), erased_iter_non_region[:,0].view(-1,1).repeat(1,3).view(-1,1)), dim=1)
        faces_to_compress = torch.cat((faces_to_compress, to_add)).unique(dim=0)
        # faces_to_compress_rest = faces_to_compress.clone().type(torch.int).to(device)
        
        cpt, indices_cpt = verify_compatibility(list_ids[torch.isin(indices_2[list_ids],meshes_continue_pooling)],indices_2[list_ids[torch.isin(indices_2[list_ids],meshes_continue_pooling)]],faces,adj_face_1,adj_face_2,regions,erased_total,faces_to_compress,indices_2)

        erased_iter_2 = erased_total[torch.isin(erased_total[:,0], cpt)]
        erased_iter_2[:,0] = indices_2[erased_iter_2[:,0]]
        erased_iter_2 = erased_iter_2[erased_iter_2[:,1] != -1]
        unique,counts = erased_iter_2[:,[0,2]].unique(dim=0,return_counts=True)

        nb_faces_per_mesh = torch.zeros(meshes_continue_pooling.size(0)).type(torch.int).to(regions.device)
        for i in range(meshes_continue_pooling.size(0)):
            counts_i = counts[unique[:,0] == meshes_continue_pooling[i]]
            idx0, = torch.where((num_faces[i] - counts_i.cumsum(dim=0)) < target_size)
            if idx0.size(0)>0:
                nb_faces_per_mesh[i] = idx0[0] + 1
            else:
                nb_faces_per_mesh[i] = counts_i.size(0)
        
        unique,counts = indices_cpt.unique(return_counts=True)
        head_meshes_2 = torch.cat((torch.zeros(1).type(torch.int).to(device), counts.cumsum(dim=0)[:-1]))

        list_regions_retained = torch.cat([cpt[head_meshes_2[j]:head_meshes_2[j]+nb_faces_per_mesh[j]] for j in range(meshes_continue_pooling.size(0))]).type(torch.int).to(device)

        ids_removed = torch.isin(erased_total[:,0],list_regions_retained)
        
        erased_iter = erased_total[ids_removed]
        erased_iter[:,0] = indices_2[erased_iter[:,0]]

        if list_regions_retained.size(0)>0:
            adj_face_1_glob = adj_face_1[:,list_regions_retained]
            adj_face_2_glob = adj_face_2[:,list_regions_retained]
            regions_glob = regions[:,:,list_regions_retained,:]
            indices = indices_2[list_regions_retained]
            faces_to_compress = faces_to_compress[torch.isin(faces_to_compress[:,1],list_regions_retained)]

            map = (torch.ones(list_regions_retained.max()+1)*-1).to(device).type(torch.int)
            map[list_regions_retained] = torch.arange(list_regions_retained.shape[0]).to(device).type(torch.int)

            faces_to_compress[:,1] = map[faces_to_compress[:,1]]
            
            idx_glob = ids[torch.isin(ids[:,0],list_regions_retained)]
            idx_glob = torch.cat((idx_glob,map[idx_glob[:,0]].unsqueeze(1)),dim=1)
            idx_glob[:,0] = indices_2[idx_glob[:,0]]

            erased_iter[:,2] = map[erased_iter[:,2]]
            idx_glob = torch.cat((idx_glob,erased_iter), dim=0).unique(dim=0)
            idx_glob = idx_glob[idx_glob[:,1]!=-1]
            return [adj_face_1_glob, adj_face_2_glob, regions_glob, faces_to_compress, erased_iter[:,:2].unique(dim=0), idx_glob, indices, list_regions_retained, meshes_not_poolable]
        else:
            return [meshes_not_poolable]
    else: return [torch.Tensor().to(device)]

def verify_compatibility(cpt,indices_cpt,faces,adj_face_1,adj_face_2,regions,erased_regions, faces_to_compress,indices_2):
    # tic=time.time()
    temp0 = torch.cat((indices_cpt.view(-1,1).repeat(3,1),adj_face_1[:,cpt].view(-1,1)),dim=1)
    temp1 = torch.cat((indices_cpt.view(-1,1).repeat(3,1),adj_face_2[:,cpt].view(-1,1)),dim=1)
    temp2 = erased_regions[erased_regions[:,1] != -1]
    temp2[:,0] = indices_2[temp2[:,0]]

    temp = torch.cat((temp2[:,:2], temp0, temp1))
    unique1,inverse1,counts1 = temp.unique(dim=0, return_counts = True, return_inverse = True)
    idx251, = torch.where(torch.isin(inverse1[:temp2.size(0)], inverse1[temp2.size(0):]))
    idx252, = torch.where(torch.isin(inverse1[temp2.size(0):], inverse1[:temp2.size(0)]))
    unique0,inverse0,counts0 = inverse1[idx251].unique(return_counts = True, return_inverse = True)
    unique2,inverse2 = inverse1[idx252 + temp2.size(0)].unique(return_inverse = True)
    cumsum = counts0.cumsum(dim=0)
    if cumsum.size(0)>0:
        start = torch.cat((torch.zeros(1).type(torch.int).to(faces.device), cumsum[:-1]))
        starts = start[inverse2]
        ends   = cumsum[inverse2]
        lengths = ends - starts

        total_len = lengths.sum()
        repeated_starts = torch.repeat_interleave(starts, lengths)
        steps = torch.arange(total_len, device=starts.device) - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        indices = repeated_starts + steps
    else: 
        indices = torch.Tensor().type(torch.int).to(faces.device)
    conflicts1 = torch.cat((indices_cpt[idx252 % cpt.size(0)].repeat_interleave(counts0[inverse2]).unsqueeze(1),
                            cpt[idx252 % cpt.size(0)].repeat_interleave(counts0[inverse2]).unsqueeze(1),
                            temp2[idx251[inverse0.sort()[1]],-1][indices].unsqueeze(1)), dim=1)

    faces_sector = faces_to_compress[torch.isin(faces_to_compress[:,1],cpt)]

    unique0 = faces_sector.unique(dim=0)
    unique0 = torch.cat((unique0,indices_2[unique0[:,1]].unsqueeze(1)),dim=1)
    unique,inverse,counts = unique0[:,[0,2]].unique(dim=0,return_inverse = True,return_counts=True)
    idx250, = torch.where(counts>1)
    idx251, = torch.where(torch.isin(inverse,idx250))
    
    cumsum = counts.cumsum(dim=0)
    if cumsum.size(0)>0:
        start = torch.cat((torch.zeros(1).type(torch.int).to(faces.device), cumsum[:-1]))
        starts = start[inverse[idx251]]
        ends   = cumsum[inverse[idx251]]
        lengths = ends - starts

        total_len = lengths.sum()
        repeated_starts = torch.repeat_interleave(starts, lengths)
        steps = torch.arange(total_len, device=starts.device) - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        indices = repeated_starts + steps
    else: 
        indices = torch.Tensor().type(torch.int).to(faces.device)

    conflicts2 = torch.cat((indices_2[unique0[idx251,1]].repeat_interleave(counts[inverse[idx251]]).unsqueeze(1), unique0[idx251,1].repeat_interleave(counts[inverse[idx251]]).unsqueeze(1), unique0[indices,1].unsqueeze(1)),dim=1)
    conflicts2 = conflicts2[conflicts2[:,1] != conflicts2[:,2]]
    conflicts = torch.cat((conflicts1,conflicts2))

    _, idx = conflicts[:,1:].sort(dim=1)
    conflicts[:,1:] = conflicts[:,1:][torch.arange(conflicts.size(0))[:,None],idx]
    conflicts = conflicts.unique(dim=0)

    temp = find_conflicts_equal(conflicts)
    idx0, idx1 = temp[:,0],temp[:,1]
    
    while idx0.size(0)>0:
        idx_temp = idx1[idx1 == idx0]
        # idx100,idx200 =(torch.where(idx0 == idx1.unsqueeze(1)))
        idx200, = torch.where(torch.isin(idx0, idx1))
        mask = torch.ones(idx0.size(0),dtype=bool)
        mask[idx200] = False
        idx0,idx1 = idx0[mask],idx1[mask]
        mask = torch.ones(conflicts.size(0),dtype=bool)
        mask[idx1] = False
        mask[idx_temp] = False
        conflicts = conflicts[mask]

        temp = find_conflicts_equal(conflicts)
        idx0, idx1 = temp[:,0],temp[:,1]
        idx0,idx1 = idx0[idx0!=idx1], idx1[idx0!=idx1]

    temp1 = torch.cat((indices_cpt.unsqueeze(1),cpt.unsqueeze(1)),dim=1)
    temp = torch.cat((temp1,conflicts[:,[0,2]]))
    unique30,inverse30,counts30 = temp.unique(dim=0,return_counts = True, return_inverse = True)
    idx250, = torch.where(counts30 == 1)
    idx251, = torch.where(torch.isin(inverse30[:temp1.size(0)], idx250))
    indices_cpt,cpt = temp1[idx251,0], temp1[idx251,1]
    
    return cpt, indices_cpt

def find_conflicts_equal(conflicts):
    temp = torch.cat((conflicts[:,[0,1]],conflicts[:,[0,2]]))
    unique,inverse = temp.unique(dim=0,return_inverse = True)
    idx250, = torch.where(torch.isin(inverse[:conflicts.size(0)], inverse[conflicts.size(0):]))
    idx251, = torch.where(torch.isin(inverse[conflicts.size(0):], inverse[:conflicts.size(0)]))
    unique0,inverse0,counts0 = inverse[idx250].unique(return_counts = True, return_inverse = True)
    unique2,inverse2 = inverse[idx251 + conflicts.size(0)].unique(return_inverse = True)
    cumsum = counts0.cumsum(dim=0)
    if cumsum.size(0)>0:
        start = torch.cat((torch.zeros(1).type(torch.int).to(conflicts.device), cumsum[:-1]))
        starts = start[inverse2]
        ends   = cumsum[inverse2]
        lengths = ends - starts

        total_len = lengths.sum()
        repeated_starts = torch.repeat_interleave(starts, lengths)
        steps = torch.arange(total_len, device=starts.device) - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        indices = repeated_starts + steps
    else: 
        indices = torch.Tensor().type(torch.int).to(conflicts.device)
    temp = torch.cat((idx251.repeat_interleave(counts0[inverse2]).unsqueeze(1),
                            idx250[inverse0.sort()[1]][indices].unsqueeze(1)), dim=1)
    return temp

def update_faces(data, faces, indices, regions_glob, faces_to_compress, adjacency_matrix, features, erased_iter, idx_glob, adj_face_1, adj_face_2, inverse_inds1, inverse_inds2):
    device = data[0].device
    replacement = faces[indices,regions_glob[0,0,:,0],0]

    # Erase the faces that are erased during the pooling to reduce the number of operations
    faces[erased_iter[:,0], erased_iter[:,1]] = -1
    # faire en sorte que ce ne soit pas forcément au même endroit dans les 3 valeurs
    idx0, idx1, idx2, idx3, idx4 = torch.Tensor().to(device).type(torch.int),torch.Tensor().to(device).type(torch.int),torch.Tensor().to(device).type(torch.int),torch.Tensor().to(device).type(torch.int), torch.Tensor().to(device).type(torch.int)
    for i in range(faces.size(0)):
        idx10, = torch.where(indices == i)
        indice_faces_comp, = torch.where(torch.isin(faces_to_compress[:,1],idx10))
        faces_to_compress_i = faces_to_compress[indice_faces_comp]

        idx_temp1, idx_temp2 = utils.where_equality_2_sides(faces[i].view(-1), faces_to_compress_i[:,0])
        idx1_0, idx2_0, idx3_0 = idx_temp2 // 3, idx_temp1, idx_temp2 % 3

        idx1, idx2, idx3 = torch.cat([idx1,idx1_0]), torch.cat([idx2,faces_to_compress_i[idx2_0,1]]), torch.cat([idx3,idx3_0])
        idx4 = torch.cat([idx4,indice_faces_comp[idx2_0]])
        idx0 = torch.cat([idx0,torch.ones(idx1_0.size(0)).to(device).type(torch.int) * i])

    # print(idx0[idx1 == 2148], idx1[idx1 == 2148], idx3[idx1 == 2148], idx2[idx1 == 2148])
    idx_cat,inverse_0 = torch.cat((idx0.unsqueeze(1),idx1.unsqueeze(1),idx3.unsqueeze(1),idx2.unsqueeze(1)),dim=1).unique(dim=0,return_inverse=True)
    idx_cat_unique,inverse,counts = idx_cat[:,:-1].unique(dim=0, return_inverse=True,return_counts=True)

    idx_mult, = torch.where(counts > 1)   
    idx_temp_1,idx_temp_2 = torch.Tensor().type(torch.int),torch.Tensor().type(torch.int)

    to_equalize = torch.cat((idx2[idx_temp_1].unsqueeze(1),idx2[idx_temp_2].unsqueeze(1)),dim=1)
    _,inds = to_equalize.min(dim=1)
    inds, = torch.where(inds == 1)
    to_equalize[inds,0], to_equalize[inds,1] = to_equalize[inds,1], to_equalize[inds,0]
    to_equalize = to_equalize.unique(dim=0)
    
    unique,counts = to_equalize.unique(return_counts = True)
    prob = unique[counts>1]

    idx_temp_1,_,idx_temp_2 = torch.where(to_equalize.unsqueeze(1) == prob.unsqueeze(1))

    unique,counts = idx_temp_1.unique(return_counts = True)
    double_prob = unique[counts == 2]

    while double_prob.size(0) > 0:
        i=double_prob[0]
        to_change_0, to_change_2 = torch.where(to_equalize == to_equalize[i,1])

        # to_change_1 = to_change_1[to_change_0 != i]
        to_change_2 = to_change_2[to_change_0 != i]
        to_change_0 = to_change_0[to_change_0 != i]

        to_equalize[to_change_0,to_change_2] = to_equalize[i,0].clone()

        to_equalize = to_equalize[to_equalize[:,0] != to_equalize[:,1]]
        to_equalize = to_equalize.unique(dim=0)

        _,inds = to_equalize.min(dim=1)
        inds, = torch.where(inds == 1)
        to_equalize[inds,0], to_equalize[inds,1] = to_equalize[inds,1], to_equalize[inds,0]

        unique,counts = to_equalize.unique(return_counts = True)
        prob = unique[counts>1]

        idx_temp_1,_,idx_temp_2 = torch.where(to_equalize.unsqueeze(1) == prob.unsqueeze(1))

        unique,counts = idx_temp_1.unique(return_counts = True)
        double_prob = unique[counts == 2]

    to_equalize = to_equalize.unique(dim=0)
    unique,counts = to_equalize.unique(return_counts = True)
    prob = unique[counts>1]

    idx_temp_1,_,idx_temp_2 = torch.where(to_equalize.unsqueeze(1) == prob.unsqueeze(1))

    idx_temp,= torch.where(idx_temp_2 == 1)
    to_equalize[idx_temp_1[idx_temp],0], to_equalize[idx_temp_1[idx_temp],1] = to_equalize[idx_temp_1[idx_temp],1], to_equalize[idx_temp_1[idx_temp],0]
    # print(replacement[[109,1009,1039,1099]])
    replacement[to_equalize[:,1]] = replacement[to_equalize[:,0]]
    
    faces[idx0, idx1, idx3] = replacement[idx2]
    idx40, idx41, idx42,_ = torch.where(data.edges[idx0[:,None],data.edges_adjacency[idx0,idx1]].unsqueeze(3) == faces_to_compress[idx4,0].unsqueeze(1).unsqueeze(2).unsqueeze(1))
    data.edges[idx0[idx40],data.edges_adjacency[idx0[idx40],idx1[idx40],idx41],idx42] = replacement[idx2[idx40]].type(data.edges.type())

    idx10,idx11,idx12 = torch.where(adj_face_1.unsqueeze(1) == adj_face_2.unsqueeze(0))
    unique,counts = idx12.unique(return_counts=True)

    _,idx21 = torch.where(idx12 == unique[counts==3].unsqueeze(1))
    idx100,idx101,idx102 = idx10[idx21].view(-1,3),idx11[idx21].view(-1,3),idx12[idx21][::3]
    idx100 = torch.cat((idx100.unsqueeze(2),idx101.unsqueeze(2)),dim=2)
    _,idx = idx100.sort(dim=2)
    idx100 = idx100[torch.arange(idx100.size(0))[:,None,None],torch.arange(3)[None,:,None],idx]
    mask_inverse = torch.ones(inverse_inds2.size(0),dtype=bool)
    idx200,_ = torch.where(idx100[:,0] == idx100[:,1])
    unique2,counts2 = idx200.unique(return_counts=True)
    idx200 = unique2[counts2 == 2]
    temp1,temp2 = torch.where((inverse_inds1[:,0] == idx100[idx200,0,0].unsqueeze(1)) & (inverse_inds1[:,1] == idx102[idx200].unsqueeze(1)))
    _,temp1 = temp1.unique(return_counts = True)
    if temp1.size(0)>0:
        temp1 = torch.cat((torch.zeros(1).to(device).type(torch.int), temp1[:-1].cumsum(dim=0)))
    else:temp1 = torch.Tensor().type(torch.int)
    mask_inverse[temp2[temp1]] = False

    idx200,_ = torch.where(idx100[:,0] == idx100[:,2])
    unique2,counts2 = idx200.unique(return_counts=True)
    idx200 = unique2[counts2 == 2]
    temp1,temp2 = torch.where((inverse_inds1[:,0] == idx100[idx200,0,0].unsqueeze(1)) & (inverse_inds1[:,1] == idx102[idx200].unsqueeze(1)))
    _,temp1 = temp1.unique(return_counts = True)
    if temp1.size(0)>0:
        temp1 = torch.cat((torch.zeros(1).to(device).type(torch.int), temp1[:-1].cumsum(dim=0)))
    else:temp1 = torch.Tensor().type(torch.int)
    mask_inverse[temp2[temp1]] = False

    idx200,_ = torch.where(idx100[:,1] == idx100[:,2])
    unique2,counts2 = idx200.unique(return_counts=True)
    idx200 = unique2[counts2 == 2]
    temp1,temp2 = torch.where((inverse_inds1[:,0] == idx100[idx200,1,0].unsqueeze(1)) & (inverse_inds1[:,1] == idx102[idx200].unsqueeze(1)))
    _,temp1 = temp1.unique(return_counts = True)
    if temp1.size(0)>0:
        temp1 = torch.cat((torch.zeros(1).to(device).type(torch.int), temp1[:-1].cumsum(dim=0)))
    else:temp1 = torch.Tensor().type(torch.int)
    mask_inverse[temp2[temp1]] = False

    _,idx21 = torch.where(idx12 == unique[counts==2].unsqueeze(1))
    idx10,idx11,idx12 = idx10[idx21].view(-1,2),idx11[idx21].view(-1,2),idx12[idx21].view(-1,2)[:,0]
    
    unique,counts = idx12.unique(return_counts = True)
    _,idx = idx10.sort(dim=1)
    idx10 = idx10[torch.arange(idx10.size(0))[:,None],idx]
    _,idx = idx11.sort(dim=1)
    idx11 = idx11[torch.arange(idx11.size(0))[:,None],idx]
    idx_rem,_ = torch.where(idx10 != idx11)
    mask = torch.ones(idx12.size(0),dtype=bool)
    mask[idx_rem] = False
    idx10,idx12 = idx10[mask][:,0],idx12[mask]

    temp1,temp2 = torch.where((inverse_inds1[:,0] == idx10.unsqueeze(1)) & (inverse_inds1[:,1] == idx12.unsqueeze(1)))
    _,temp1 = temp1.unique(return_counts = True)
    if temp1.size(0)>0:
        temp1 = torch.cat((torch.zeros(1).to(device).type(torch.int), temp1[:-1].cumsum(dim=0)))
    else:temp1 = torch.Tensor().type(torch.int)
    mask_inverse[temp2[temp1]] = False
    inverse_inds1,inverse_inds2 = inverse_inds1[mask_inverse],inverse_inds2[mask_inverse]

    idx10,idx11 = torch.where(adj_face_1 == -1)

    _,idx12 = torch.where((inverse_inds2[:,0] == idx10.unsqueeze(1)) & (inverse_inds2[:,1] == idx11.unsqueeze(1)))
    mask = torch.ones(inverse_inds2.size(0), dtype=bool)
    mask[idx12] = False
    inverse_inds1,inverse_inds2 = inverse_inds1[mask],inverse_inds2[mask]

    data.edges_adjacency[indices[inverse_inds2[:,1]], adj_face_2[inverse_inds2[:,0],inverse_inds2[:,1]], inverse_inds2[:,2]] = data.edges_adjacency[indices[inverse_inds2[:,1]], adj_face_1[inverse_inds2[:,0],inverse_inds2[:,1]], inverse_inds1[:,2]]
    
    data.edges_adjacency[erased_iter[:,0],erased_iter[:,1]] = -1
    edges_erase = torch.arange(data.edges.size(1)).unsqueeze(0).repeat(faces.size(0),1)

    edges_erase[torch.arange(faces.size(0))[:,None], data.edges_adjacency[:,:,0]] = -1
    edges_erase[torch.arange(faces.size(0))[:,None], data.edges_adjacency[:,:,1]] = -1
    edges_erase[torch.arange(faces.size(0))[:,None], data.edges_adjacency[:,:,2]] = -1
    edges_erase[:,-1] = 1
    
    idx0,idx1 = torch.where(edges_erase!=-1)
    edges_to_erase = torch.cat((idx0.unsqueeze(1), idx1.unsqueeze(1)),dim=1)

    data.edges_erased = edges_to_erase.to(data.device)
    data.erase_align_edges()
    # Décaler edges
    data.edges[data.edges_erased[:,0],data.edges_erased[:,1]] = -1

    unique1 = data.edges[0].unique()
    unique2 = data.faces[0].unique()

    return faces, adjacency_matrix, features, erased_iter