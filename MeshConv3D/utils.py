import torch
import numpy as np
import torch.nn as nn
# import pandas as pd
from typing import List
import time
import trimesh
import trimesh as tm
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import mesh
from scipy.spatial import KDTree
import os
import subprocess
import prepare_dataset
# import my_lib_cuda
# from nndistance.modules.nnd import NNDModule

draco_encoder = "/home/germain_faster/PHD_work/draco/build_dir/draco_encoder"
draco_decoder = "/home/germain_faster/PHD_work/draco/build_dir/draco_decoder"

def fix_unknown_connectivity(mesh, adjacent_faces, face_adjacency_edges, faces_wrong):
    idx0,idx1,_ = torch.where(torch.Tensor(mesh.face_adjacency) == faces_wrong.unsqueeze(1).unsqueeze(2))

    edges_faces_wrong = torch.Tensor(mesh.faces_unique_edges).type(torch.int)[faces_wrong]

    idx2,idx3 = torch.where((face_adjacency_edges[idx1,0].unsqueeze(1) == torch.Tensor(mesh.edges_unique[edges_faces_wrong[idx0],0])) &
                (face_adjacency_edges[idx1,1].unsqueeze(1) == torch.Tensor(mesh.edges_unique[edges_faces_wrong[idx0],1])))

    mask = torch.ones((faces_wrong.size(0),3),dtype=bool).view(-1,3)
    mask[idx0,idx3] = False

    solver = torch.cat((faces_wrong.unsqueeze(1).repeat(1,3)[mask].unsqueeze(1), edges_faces_wrong[mask].unsqueeze(1)), dim=1)
    _,idx = solver[:,1].sort()
    solver = solver[idx]

    unique,counts = solver[:,1].unique(return_counts=True)

    idx10, = torch.where(counts % 2 == 1)
    idx_keep = torch.arange(solver.size(0))
    _,idx11 = torch.where(solver[:,1] == unique[idx10].unsqueeze(1))
    if idx10.size(0)>0:
        head_rem = torch.cat((torch.zeros(1).type(torch.int), counts[idx10].cumsum(dim=0)[:-1]))
    else: head_rem = torch.Tensor().type(torch.int)
    mask = torch.ones(idx_keep.size(0),dtype=bool)
    mask[idx11[head_rem]] = False
    idx_keep = idx_keep[mask]

    solver_void = solver[idx11[head_rem]]#torch.isin(solver[:,1], unique[counts!=1], invert = True)
    solver = solver[idx_keep]

    solver_void = torch.cat((solver_void, (torch.ones(solver_void.size(0)) * 2).type(torch.int).unsqueeze(1)),dim=1)
    unique,counts = solver_void[:,0].unique(return_counts=True)
    _,idx1 = torch.where(solver_void[:,0] == unique[counts == 2].unsqueeze(1))
    solver_void[idx1[::2],2] = 1

    adjacent_faces = torch.cat((adjacent_faces, solver[:,0].view(-1,2)))
    face_adjacency_edges = torch.cat((face_adjacency_edges, torch.Tensor(mesh.edges_unique[solver[::2,-1]]).view(-1,2))).type(torch.int)
    return adjacent_faces, face_adjacency_edges, solver_void

import numpy as np
import torch

def create_mesh_adj_matrix(mesh,common_edge=True):
    """
    Create the face adjacency matrix for the computation of the ring-K and to save the connectivity
    The faces are added in order for each face. The order is defined by the face area of each face.
    """
    adjacent_faces = torch.Tensor(mesh.face_adjacency)
    face_adjacency_edges = torch.Tensor(mesh.face_adjacency_edges).type(torch.int)
    num_faces = mesh.faces.shape[0]
    adj_matrix = (np.ones((num_faces+1, 3)) * -1).astype(int)

    unique,counts = torch.Tensor(mesh.face_adjacency).type(torch.int).unique(return_counts=True)
    missing = torch.arange(num_faces)
    mask = torch.ones(num_faces,dtype=bool)
    mask[unique] = False
    missing = missing[mask]
    to_correct = torch.cat((unique[counts!=3],missing)).unique()

    if to_correct.size(0) > 0:
        adjacent_faces, face_adjacency_edges, solver_void = fix_unknown_connectivity(mesh, adjacent_faces, face_adjacency_edges, to_correct)
    else: solver_void = torch.Tensor().view(-1,3).type(torch.int)

    temp = torch.Tensor(adjacent_faces).type(torch.int)
    temp = torch.cat((temp, temp[:,[1,0]]))
    unique,counts = torch.Tensor(temp[:,0]).unique(return_counts=True)
    temp = torch.cat((temp, torch.cat((unique[counts == 2].unsqueeze(1), (torch.ones(unique[counts==2].size(0)) * -1).unsqueeze(1).type(torch.int)),dim=1)))
    temp = torch.cat((temp, torch.cat((unique[counts == 1].unsqueeze(1).repeat(2,1), (torch.ones(unique[counts==1].size(0)) * -1).unsqueeze(1).type(torch.int).repeat(2,1)),dim=1)))

    _,idx = temp[:,0].sort(dim=0)
    temp = temp[idx]
    faces_2 = temp[:,1].view(-1,3).type(torch.int)
    idx[idx >= adjacent_faces.shape[0]] = idx[idx >= adjacent_faces.shape[0]] - adjacent_faces.shape[0]

    if not common_edge:
        area_faces = np.concatenate((mesh.area_faces,np.ones(1) * -1))
        faces_area_sorted_2 = area_faces[faces_2].argsort(axis=1)[:,::-1]
    else:
        idx[temp[:,1] == -1] = -1
        edges_adjacency = idx.view(-1,3)

        edges_length = torch.norm(torch.Tensor(mesh.vertices)[face_adjacency_edges[:,0]] - torch.Tensor(mesh.vertices)[face_adjacency_edges[:,1]], dim=1)
        edges_length = torch.cat((edges_length, torch.ones(1) * torch.inf))

        faces_area_sorted_2 = edges_length[edges_adjacency]
        faces_area_sorted_2 = faces_area_sorted_2.argsort(axis=1)

    adj_matrix[:num_faces] = faces_2[np.arange(num_faces)[:,None],faces_area_sorted_2]
    edges_adjacency = edges_adjacency[np.arange(num_faces)[:,None],faces_area_sorted_2]
    edges_adjacency[solver_void[:,0], solver_void[:,2]] = torch.arange(face_adjacency_edges.size(0), face_adjacency_edges.size(0) + solver_void.size(0))
    face_adjacency_edges = torch.cat((face_adjacency_edges, torch.Tensor(mesh.edges_unique[solver_void[:,1]]).view(-1,2))).type(torch.int)

    return adj_matrix, edges_adjacency, torch.cat((face_adjacency_edges, (torch.ones((1,2))*-1).type(torch.int)))

def average_pooling_after_removing_useless_faces(x_2, erased):
    ret_tensor = torch.Tensor(x_2[1].shape[0], x_2[1].shape[2])
    for i in range(x_2[1].shape[0]):
        # idx = torch.where(x_2[2][i] == -1)
        idxs = erased[erased[:,0] == i][:,1]

        mask = torch.ones(x_2[1][i].numel(), dtype=torch.bool).view(x_2[1].shape[1], x_2[1].shape[2])
        mask[idxs] = False
        mesh = x_2[1][i][mask].view(-1, x_2[1].shape[2])
        # ret_tensor[i] = nn.AvgPool1d(mesh.shape[0])(mesh.T).T
        ret_tensor[i] = nn.MaxPool1d(mesh.shape[0])(mesh.T).T

    ret_tensor = ret_tensor.clone().to(x_2[1].device)
    ret_tensor[ret_tensor.isnan()] = 0
    ret_tensor[ret_tensor.isinf()] = 0
    ret_tensor = ret_tensor.requires_grad_()

    return ret_tensor


def collate_fn_dft(batch, meshes_only = False):
    if meshes_only:
        mesh_features = list(zip(*batch))
        descriptors = True
        files,label,idx = [],[],[]
    else: 
        mesh_features, descriptors, files, label, idx = zip(*batch)
        mesh_features = list(zip(*mesh_features))

    meshes = mesh.Mesh_Tensor(descriptors)
    meshes.files = torch.Tensor(list(files)).type(torch.long)
    
    indices = list(idx)

    faces = mesh_features[0]
    if descriptors:
        edge_features, face_features = mesh_features[1], mesh_features[2]
        i=3
    else:
        face_features = mesh_features[1]
        i=2
        
    adjacency_matrix = mesh_features[i]
    nb_faces = mesh_features[i+1]
    vertices = mesh_features[i+2]
    edges_adjacency = mesh_features[i+3]
    edges = mesh_features[i+4]

    N = len(batch)
    meshes.N = N
    max_faces = max(nb_faces)
    min_faces = min(nb_faces)

    batch_faces = torch.ones((N, max_faces+1, 3)) * -1
    batch_faces[:,-1] = -1
    if edge_features[0].shape[0]>0:
        batch_edge_feats = torch.zeros((N, max_faces+1, edge_features[0].shape[1], edge_features[0].shape[2]))
        batch_face_features = torch.zeros((N, max_faces+1, face_features[0].shape[1], face_features[0].shape[2]))
    else:
        batch_face_features = torch.zeros((N, max_faces+1, face_features[0].shape[1]))#*face_features[0].shape[2]))
    batch_adjacency = torch.ones((N, max_faces+1, 3)) * -1
    batch_adjacency[:,-1] = -1
    batch_edge_adjacency = torch.ones((N, max_faces+1, 3)) * -1
    batch_edge_adjacency[:,-1] = -1
    batch_ring_k = torch.Tensor()
    label = torch.Tensor(label).type(torch.long)

    nb_points = np.array([i.shape[0] for i in vertices])
    batch_vertices = torch.zeros(N, nb_points.max(), 3)
    nb_edges = np.array([i.shape[0] for i in edges])
    batch_edges = torch.ones(N, nb_edges.max(), 2) * -1

    erased = (torch.ones((N,max_faces-min_faces+1)) * -1).type(torch.int)
    erased[:,0] = torch.ones(N) * (max_faces)
    erased_edges = (torch.ones((N,nb_edges.max()-nb_edges.min()+1)) * -1).type(torch.int)

    for i in range(N):
        erased[i,1:(max_faces - nb_faces[i]+1)] = torch.arange(nb_faces[i], max_faces)
        batch_faces[i,:nb_faces[i]] = torch.Tensor(faces[i])
        if edge_features[0].shape[0]>0:
            batch_edge_feats[i,:nb_faces[i]] = torch.Tensor(edge_features[i][:-1])
            batch_face_features[i,:nb_faces[i]] = torch.Tensor(face_features[i][:-1])
        else:
            batch_face_features[i,:nb_faces[i]] = torch.Tensor(face_features[i])#.view(-1,12)
        batch_adjacency[i,:nb_faces[i]] = torch.Tensor(adjacency_matrix[i][:-1])
        batch_edge_adjacency[i,:nb_faces[i]] = torch.Tensor(edges_adjacency[i])
        batch_vertices[i,:nb_points[i]] = torch.Tensor(vertices[i])
        batch_edges[i,:nb_edges[i]] = torch.Tensor(edges[i])
        erased_edges[i,1:(nb_edges.max() - nb_edges[i]+1)] = torch.arange(nb_edges[i], nb_edges.max())

    erased = torch.cat((torch.arange(batch_faces.shape[0]).unsqueeze(1).repeat(1,erased.shape[1]).view(-1,1), erased.view(-1,1)),dim=1)
    erased = erased[erased[:,1] != -1]
    erased_edges = torch.cat((torch.arange(batch_faces.shape[0]).unsqueeze(1).repeat(1,erased_edges.shape[1]).view(-1,1), erased_edges.view(-1,1)),dim=1)
    erased_edges = erased_edges[erased_edges[:,1] != -1]

    meshes.faces = batch_faces.type(torch.long)
    if edge_features[0].shape[0]>0:
        meshes.edge_features = batch_edge_feats.type(torch.float)
    meshes.face_features = batch_face_features.type(torch.float)
    meshes.adjacency = batch_adjacency.type(torch.long)
    meshes.erased = erased.type(torch.int)
    meshes.vertices = batch_vertices.type(torch.float)
    meshes.edges_adjacency = batch_edge_adjacency.type(torch.int)
    meshes.edges = batch_edges.type(torch.int)
    meshes.edges_erased = erased_edges.type(torch.int)
    
    return meshes, label, indices

def remove_zeros(x, num_faces, erased, device, points=False,unpool=False):
    num_faces_to_remove = x[1].shape[1] - num_faces
    to_remove = torch.zeros((x[1].shape[0],num_faces_to_remove)).type(torch.int).to(device)
    to_keep = torch.zeros((x[1].shape[0],num_faces)).type(torch.int).to(device)
    differences = torch.zeros((x[1].shape[0],x[1].shape[1])).type(torch.int).to(device)
    mask_erased = torch.ones(erased.shape[0],dtype=bool).to(device)
    idx, = torch.where(erased[:,1] == x[1].shape[1]-1)
    mask_erased[idx] = False
    
    for i in range(x[0].shape[0]):
        idx0, = torch.where(erased[:,0] == i)
        idx0 = idx0[:num_faces_to_remove-1]
        mask_erased[idx0] = False

        to_remove[i] = torch.cat((erased[idx0,1],torch.ones(1).type(torch.int).to(device) * (x[1].shape[1] - 1)))

        mask = torch.ones(x[1].shape[1], dtype=bool)
        mask[to_remove[i]] = False

        to_keep[i] = torch.arange(x[1].shape[1])[mask].type(torch.int).to(device)

        mask = torch.zeros(x[1].shape[1], dtype=bool)
        mask[to_remove[i]] = True
        
        differences[i] = torch.cumsum(mask,0)

    x[3][torch.arange(x[1].shape[0])[:,None],-1,0] = x[3][torch.arange(x[1].shape[0])[:,None],-1,0] - differences[torch.arange(x[1].shape[0])[:,None],x[3][torch.arange(x[1].shape[0])[:,None],-1,0]]
    differences[:,-1] = 0
    
    erased[mask_erased,1] = erased[mask_erased,1] - differences[erased[mask_erased][:,0],erased[mask_erased][:,1]]
    erased = erased[mask_erased]
    erased = torch.cat((erased, torch.cat((torch.arange(x[1].shape[0]).unsqueeze(1).to(device), torch.ones(x[1].shape[0]).to(device).unsqueeze(1) * num_faces), dim=1)), dim=0).unique(dim=0).type(torch.int)

    x[0] = x[0][torch.arange(x[1].shape[0])[:,None], to_keep]
    x[1] = x[1][torch.arange(x[1].shape[0])[:,None], to_keep]
    x[2] = x[2][torch.arange(x[1].shape[0])[:,None], to_keep] - differences[torch.arange(x[1].shape[0])[:,None,None],x[2][torch.arange(x[1].shape[0])[:,None], to_keep]]

    x.edges_adjacency = x.edges_adjacency[torch.arange(x[1].shape[0])[:,None], to_keep]

    x[0] = torch.cat((x[0], (torch.ones(x[0].shape[0], 1, x[0].shape[2]) * -1).to(device).type(torch.long)), dim=1)
    if points:
        x[1] = torch.cat((x[1], (torch.zeros(x[1].shape[0], 1, x[1].shape[2], x[1].shape[3]).to(device)).type(torch.float)), dim=1).type(torch.float)
    else:
        x[1] = torch.cat((x[1], (torch.zeros(x[1].shape[0], 1, x[1].shape[2]).to(device)).type(torch.float)), dim=1).type(torch.float)
    x[2] = torch.cat((x[2], (torch.ones(x[2].shape[0], 1, x[2].shape[2]) * -1).to(device)), dim=1).type(torch.long)
    x[2][erased[:,0], erased[:,1]] = -1

    x[3] = x[3][torch.arange(x[1].shape[0])[:,None], to_keep] - differences[torch.arange(x[1].shape[0])[:,None,None],x[3][torch.arange(x[1].shape[0])[:,None], to_keep]]
    x[3] = torch.cat((x[3], (torch.ones(x[3].shape[0], 1, x[3].shape[2]) * -1).to(device)), dim=1).type(torch.long)
    # if not unpool:
    x.edges_adjacency = torch.cat((x.edges_adjacency, (torch.ones(x.edges_adjacency.shape[0], 1, x.edges_adjacency.shape[2]) * -1).to(device)), dim=1).type(torch.long)

    non_erased = torch.ones(x[2].size(0),x[2].size(1)).to(device)
    non_erased[erased[:,0],erased[:,1]] = -1
    _,map = non_erased.sort(dim = 1,descending=True, stable=True)
    _,inverse_map = map.sort(dim=1, stable=True)

    erased[:,1] = inverse_map[erased[:,0],erased[:,1]]
    x[0] = x[0][torch.arange(x[0].size(0))[:,None],map]
    x[1] = x[1][torch.arange(x[0].size(0))[:,None],map]
    x[2] = x[2][torch.arange(x[0].size(0))[:,None],map]
    inverse_map[:,-1] = -1
    x[2] = inverse_map[torch.arange(x[2].size(0))[:,None,None],x[2]]

    x.edges_adjacency = x.edges_adjacency[torch.arange(x[0].size(0))[:,None],map]
    return x, erased

def save_obj(vertices: torch.Tensor, faces: torch.Tensor, path: str):
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def load_obj_as_tensors(filepath):
    mesh = trimesh.load(filepath, process=True)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"File {filepath} does not contain a valid mesh.")

    # Vertices: (num_vertices, 3)
    vertices = torch.Tensor(mesh.vertices, dtype=torch.float32)

    # Faces: (num_faces, 3) -- indices into vertices
    faces = torch.Tensor(mesh.faces, dtype=torch.int32)

    return vertices, faces

def process_mesh(index, vertices_i, faces_i, mesh_device, mesh_erased_i, training):    
    # Filtrer les faces
    keep_faces = torch.isin(torch.arange(faces_i.size(0)).to(mesh_device), mesh_erased_i, invert=True)
    valid_faces = faces_i[keep_faces]

    temp_obj = f"compression/input_meshes/temp_{mesh_device}_{index}.obj"
    temp_drc = f"compression/enc_meshes/temp_{mesh_device}_{index}.drc"
    temp_decoded = f"compression/output_meshes/temp_{mesh_device}_{index}.obj"

    save_obj(vertices_i[:-1], valid_faces, temp_obj)

    subprocess.run([
        draco_encoder,
        "-i", temp_obj,
        "-o", temp_drc,
        "-cl", "10",
        "-qp", "10",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    compressed_size = os.path.getsize(temp_drc) * 8

    if not training:
        subprocess.run([
            draco_decoder,
            "-i", temp_drc,
            "-o", temp_decoded,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        mesh_uncomp = trimesh.load(temp_decoded, process=True)
        print(f"start: {temp_decoded}, rank {mesh_device}")
        mesh_features = prepare_dataset.load_mesh_without_desc(mesh_uncomp, request=['points'])
        print(f"done: {temp_decoded} , rank {mesh_device}\n")
    else:
        mesh_features = []

    return compressed_size, mesh_features

def export_mesh(vertices,faces,path_export):
    mesh = tm.base.Trimesh(vertices.clone().cpu().detach().numpy(),faces.clone().cpu().detach().numpy())
    mesh.export(path_export)

def error_meshes_mm(a, b):
    distances = torch.sqrt(torch.sum((a - b) ** 2, dim=2))

    return distances

def nndistance_simple(rec, data):
    """
    A simple nearest neighbor search, not very efficient, just for reference
    """
    rec_sq = torch.sum(rec*rec, dim=2, keepdim=True) # (B,N,1)
    data_sq = torch.sum(data*data, dim=2, keepdim=True) # (B,M,1)
    cross = torch.matmul(data, rec.permute(0, 2, 1)) # (B,M,N)
    dist = data_sq - 2 * cross + rec_sq.permute(0, 2, 1) # (B,M,N)
    data_dist, data_idx = torch.min(dist, dim=2)
    rec_dist, rec_idx = torch.min(dist, dim=1)
    return data_dist, rec_dist, data_idx, rec_idx

def chamfer(p1, p2, n1, n2):
    #p1 = rec, p2 = data
    data_dist, rec_dist, data_idx, rec_idx = nndistance_simple(p1.unsqueeze(0), p2.unsqueeze(0))
    data_dist, rec_dist = torch.mean(data_dist, 1), torch.mean(rec_dist, 1)
    loss_pos = torch.mean(data_dist + rec_dist)
    # loss_pos = (data_dist + rec_dist)
    # print((p2 - p1[data_idx[0]]).norm(p=2,dim=1)**2)

    # Normal error computation
    n1 = torch.Tensor(n1.copy())
    n2 = torch.Tensor(n2.copy())

    normal_errors_1 = torch.mean((1 - nn.functional.cosine_similarity(n1, n2[rec_idx[0].cpu()], dim=1)))
    normal_errors_2 = torch.mean((1 - nn.functional.cosine_similarity(n2, n1[data_idx[0].cpu()], dim=1)))
    # normal_errors_1 = torch.mean((1 - nn.functional.cosine_similarity(n1, n2, dim=1)))
    # normal_errors_2 = torch.mean((1 - nn.functional.cosine_similarity(n2, n1, dim=1)))
    normal_error = normal_errors_1 + normal_errors_2
    return loss_pos, normal_error

def compute_covariance_matrix(neighbors,p):
    """Compute the covariance matrix of a set of neighbor points."""
    # mean = np.mean(neighbors, axis=0)
    centered = neighbors - p

    return torch.matmul(centered.T, centered) / len(neighbors)

def estimate_curvature(points, k=10):
    """Estimate curvature for each point in the point cloud."""
    tree = KDTree(points)
    curvatures = torch.zeros(len(points))
    
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=k+1)  # k+1 to include the point itself
        neighbors = points[idx[1:]]  # Exclude the point itself
        C = compute_covariance_matrix(neighbors,p)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(C)
        eigenvalues,_ = torch.sort(eigenvalues)
        
        curvatures[i] = eigenvalues[0] / torch.sum(eigenvalues[:3])  # Using Eq. (2)
    
    return curvatures

def estimate_mean_curvature(points, curvatures, k=10, h=1.0):
    """Compute the mean curvature as a Gaussian-weighted average."""
    tree = KDTree(points)
    mean_curvatures = torch.zeros(len(points))
    
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=k+1)
        neighbors = points[idx[1:]]
        weights = torch.exp(-torch.linalg.norm(neighbors - p, axis=1)**2 / h)
        
        mean_curvatures[i] = torch.sum(curvatures[idx[1:]] * weights) / torch.sum(weights)
    
    return mean_curvatures

def estimate_roughness(curvatures, mean_curvatures):
    """Compute roughness as the absolute difference between curvature and mean curvature."""
    return torch.abs(curvatures - mean_curvatures)

def compute_roughness(point_cloud, point_cloud_orig, idx_in, idx_out):
    curvatures_out = estimate_curvature(point_cloud, k=15)
    curvatures_in = estimate_curvature(point_cloud_orig, k=15)
    mean_curvatures_out = estimate_mean_curvature(point_cloud, curvatures_out,k=15)
    mean_curvatures_in = estimate_mean_curvature(point_cloud_orig, curvatures_in,k=15)
    
    # roughness = (np.mean((curvatures_in - curvatures_out)**2))*2
    roughness_in = (torch.mean((mean_curvatures_in - mean_curvatures_out[idx_out])**2))
    roughness_out = (torch.mean((mean_curvatures_in[idx_in] - mean_curvatures_out)**2))
    # print(np.sqrt(np.mean(np.abs(estimate_roughness(curvatures_in, mean_curvatures_in) - estimate_roughness(curvatures_out, mean_curvatures_out))**2)))
    
    # roughness = estimate_roughness(curvatures, mean_curvatures)

    return roughness_in + roughness_out

def where_equality_2_sides(temp0,temp1):
    temp = torch.cat((temp0,temp1))
    unique,inverse = temp.unique(dim=0,return_inverse = True)
    idx250, = torch.where(torch.isin(inverse[:temp0.size(0)], inverse[temp0.size(0):]))
    idx251, = torch.where(torch.isin(inverse[temp0.size(0):], inverse[:temp0.size(0)]))
    unique0,inverse0,counts0 = inverse[idx250].unique(return_counts = True, return_inverse = True)
    unique2,inverse2 = inverse[idx251 + temp0.size(0)].unique(return_inverse = True)
    cumsum = counts0.cumsum(dim=0)
    if cumsum.size(0)>0:
        start = torch.cat((torch.zeros(1).type(torch.int).to(temp0.device), cumsum[:-1]))
        starts = start[inverse2]
        ends   = cumsum[inverse2]
        lengths = ends - starts

        total_len = lengths.sum()
        repeated_starts = torch.repeat_interleave(starts, lengths)
        steps = torch.arange(total_len, device=starts.device) - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        indices = repeated_starts + steps
    else: 
        indices = torch.Tensor().type(torch.int).to(temp0.device)

    return idx251.repeat_interleave(counts0[inverse2]), idx250[inverse0.sort()[1]][indices]