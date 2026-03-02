import json
import random
from pathlib import Path
import torch
import os
import pyvista
import pyacvd

from torch.utils.data import Dataset

import numpy as np
import trimesh
import trimesh as tm
import pymeshlab
from scipy.spatial.transform import Rotation

import utils
import time

np.random.seed(10)

def augment_points(pts):
    # scale
    pts = pts * np.random.uniform(0.8, 1.25)

    # translation
    translation = np.random.uniform(-0.1, 0.1)
    pts = pts + translation

    return pts


def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh.vertices = rotation.apply(mesh.vertices)
    return mesh


def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh

def random_limited_disturbance(mesh: trimesh.Trimesh):
    # Select 40% of the mesh edges and one vertex of these edges randomly
    number_edges = mesh.edges_unique.shape[0]
    rng = np.random.default_rng()
    idx_edges_disturbed = rng.choice(number_edges, size=int(0.4 * number_edges), replace=False)
    idx_vertex_disturbed  = rng.choice(2, size=int(0.4 * number_edges))

    L = mesh.edges_unique_length[idx_edges_disturbed]
    idx1_1,_,idx1_2 = torch.where((torch.Tensor(mesh.face_adjacency_edges.copy()).view(-1,2,1) == torch.Tensor(mesh.edges_unique[idx_edges_disturbed][:,0].copy()).view(-1)))
    idx2_1,_,idx2_2 = torch.where((torch.Tensor(mesh.face_adjacency_edges.copy()).view(-1,2,1) == torch.Tensor(mesh.edges_unique[idx_edges_disturbed][:,1].copy()).view(-1)))
    idx3 = torch.cat((idx1_1.view(-1,1), idx1_2.view(-1,1)),dim=1)
    idx4 = torch.cat((idx2_1.view(-1,1), idx2_2.view(-1,1)),dim=1)
    _,id1 = (torch.where((idx3[:,0].unsqueeze(0) == idx4[:,0].unsqueeze(1)) & (idx3[:,1].unsqueeze(0) == idx4[:,1].unsqueeze(1))))
    _, indices = idx1_2[id1].sort()
    theta = np.ones(idx_vertex_disturbed.shape[0]) * np.pi
    theta[idx1_2[id1[indices]]] = mesh.face_adjacency_angles[idx1_1[id1[indices]]]

    len = 2*L*(theta/10)

    # Randomly disturbs the selected vertices
    len = np.resize(len,(len.shape[0],1))
    mesh.vertices[mesh.edges_unique[idx_edges_disturbed,idx_vertex_disturbed]] = mesh.vertices[mesh.edges_unique[idx_edges_disturbed,idx_vertex_disturbed]] + np.random.default_rng().uniform(-len,len,(len.shape[0],3))

    return mesh

def mesh_normalize(mesh: trimesh.Trimesh):
    # Normalizes each mesh
    vertices = mesh.vertices - mesh.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh.vertices = vertices
    return mesh

def get_adj_nm(adjs_): # integrate all the adjacent face lists to make a unified structure including each face and its three adjacent faces.
    re = torch.cat([adjs_[:,1].unsqueeze(1), adjs_[:,0].unsqueeze(1)],dim=1)
    ag = torch.cat([adjs_, re],dim=0)
    tt = ag[ag[:,0].sort()[1]]
    t1 = tt[:,0].unique(return_inverse =True)[1]
    t2 = (tt[:,0].unique(return_counts =True)[1] == 3).nonzero(as_tuple=False)[:,0].unsqueeze(1)
    uid = ((t1 == t2) == True).nonzero(as_tuple=False)[:,1]
    aa= tt[uid].reshape(-1,6)
    add=torch.unique(aa, dim=1)
    return add
    
def get_edges(face_, verts_, i): # looking for an edge path
    tf = face_[0]
    tt = face_[1:]
    edge = []
    for i in range(3):
        t2 = tf[i].unsqueeze(0).repeat(3,1)
        iid = (tt== t2).nonzero(as_tuple = False)
        pp = tt[iid[:,0]].flatten().unique(return_counts=True)[0]
        pi = ((pp != tf[0]) & (pp != tf[1]) & (pp != tf[2])).nonzero(as_tuple=False)[:,0]
        pp[pi]
        e2 = pp[pi].reshape(-1,1)
        if e2.shape[0]<2:
            edge.append(torch.zeros(2,3))
        else:
            e1 = tf[i].unsqueeze(0).repeat(e2.shape[0],1)
            e1 = verts_[e1]
            e2 = verts_[e2]
            edge.append((e1 - e2).squeeze(1))
    return torch.stack(edge,dim=0)

def load_mesh_without_desc(mesh, normalize=False, augments=[], request=[], K=4):

    if normalize:
        mesh = mesh_normalize(mesh)

    F = mesh.faces
    V = mesh.vertices
    Fs = mesh.faces.shape[0]
    adj_matrix, edges_adjacency, edges = utils.create_mesh_adj_matrix(mesh,common_edge=True)
    # ring_K = utils.compute_ring_k(adj_matrix, Fs+1, K)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    # corner = V[F.flatten()].reshape(-1, 3, 3) - face_center[:, np.newaxis, :]
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])
    
    feats = []
    if 'points' in request:
        feats.append(V[F].reshape(F.shape[0],9))
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.concatenate(feats,axis=1)

    verts = np.concatenate((mesh.vertices,torch.zeros(1,3)))
    verts_t = torch.from_numpy(verts).detach().numpy()
    
    # if 'points' in request:
    #     return mesh.faces, mesh.vertices, feats, Fs, adj_matrix#, ring_K
    return [mesh.faces, np.zeros(0), feats, adj_matrix, mesh.faces.shape[0], verts_t, edges_adjacency, edges]


def prepare_dataset_SHREC(dataset):
    """
    This functions traverses the dataset starting from the dataroot. It splits the dataset into a training and a testing set.
    According to the options, it creates a set number of augmented versions of the meshes, and extracts the features of each 
    mesh that will be used in the network. It then saves the numpy arrays so that at each iteration of the network the loading
    of the data is as efficient as possible.
    """
    dataset.shape_classes = sorted([x.name for x in dataset.dataroot.iterdir() if x.is_dir()])
    new_dataroot = str(dataset.dataroot) + f'_spit{dataset.nb_split}_{dataset.nb_augments}_{"_".join(dataset.augments)}'
    if not dataset.descriptors:
        new_dataroot = new_dataroot + '_noDesc'
    check = os.path.exists(new_dataroot)
    cpt2 = 0
    if not check:
        os.mkdir(new_dataroot)
    else: dataset.dataroot= Path(new_dataroot)
    for obj_class in dataset.dataroot.iterdir():
        if obj_class.is_dir():
            if not check:
                os.mkdir(new_dataroot + '/' + obj_class.name)
                os.mkdir(new_dataroot + '/' + obj_class.name + '/train')
                os.mkdir(new_dataroot + '/' + obj_class.name + '/test')
            label = dataset.shape_classes.index(obj_class.name)

            if not check:
                dir_iter = (obj_class).iterdir()
            else:
                dir_iter = (obj_class / dataset.mode).iterdir()

            rng = np.random.default_rng()
            train_test_split = np.concatenate((np.zeros(20-dataset.nb_split), np.ones(dataset.nb_split))).astype(int)
            rng.shuffle(train_test_split)
            cpt=0

            for obj_path in dir_iter:
                if obj_path.is_file():
                    if not check:
                        train = train_test_split[cpt]
                        if train:
                            for i in range(dataset.nb_augments):
                                mesh = trimesh.exchange.load.load_mesh(obj_path,file_type='obj')

                                for method in dataset.augments:
                                    if method == 'orient':
                                        mesh = randomize_mesh_orientation(mesh)
                                    if method == 'scale':
                                        mesh = random_scale(mesh)
                                    if method == 'lim_dist':
                                        mesh = random_limited_disturbance(mesh)

                                obj_name = str(obj_path.name).split('.')[0]+f'_{i}.npy'

                                
                                
                                if train:
                                    output_path = str(new_dataroot + '/' + obj_class.name + '/train/' + obj_name)
                                    dataset.mesh_paths.append(new_dataroot + '/' + obj_class.name + '/train/' + obj_name)
                                    
                                    dataset.labels.append(label)
                                else:
                                    output_path = str(new_dataroot + '/' + obj_class.name + '/test/' + obj_name)
                                    dataset.mesh_paths.append(new_dataroot + '/' + obj_class.name + '/test/' + obj_name)
                                    
                                    dataset.labels.append(label)

                                
                                mesh_features = load_mesh_without_desc(mesh, normalize=True, augments=dataset.augments, request=dataset.feats, K=dataset.K)
                                np.save(output_path, np.array(mesh_features, dtype=object))
                            
                        else:
                            mesh = trimesh.exchange.load.load_mesh(obj_path,file_type='obj')

                            obj_name = str(obj_path.name).split('.')[0]+'.npy'

                            output_path = str(new_dataroot + '/' + obj_class.name + '/test/' + obj_name)


                            mesh_features = load_mesh_without_desc(mesh, normalize=True, augments=dataset.augments, request=dataset.feats, K=dataset.K)
                            np.save(output_path, np.array(mesh_features, dtype=object))

                            if dataset.mode == 'test':
                                dataset.mesh_paths.append(new_dataroot + '/' + obj_class.name + '/test/' + obj_name)
                                
                                dataset.labels.append(label)


                        cpt += 1
                        
                    else:
                        dataset.mesh_paths.append(obj_path)
                        dataset.labels.append(label)

                        name = str(obj_path.name).split('.')[0].split('_')[0]
                        if dataset.mode == 'test' and name not in dataset.files_map:
                            dataset.files_map[name] = cpt2
                            dataset.files.append(cpt2)
                            dataset.file_label.append(label)
                            cpt2+=1
                        elif dataset.mode == 'test':
                            dataset.files.append(dataset.files_map[name])



    dataset.mesh_paths = np.array(dataset.mesh_paths)
    dataset.labels = np.array(dataset.labels)
    dataset.dataroot = new_dataroot

def prepare_dataset_Manifold(dataset):
    """
    This functions traverses the dataset starting from the dataroot. It splits the dataset into a training and a testing set.
    According to the options, it creates a set number of augmented versions of the meshes, and extracts the features of each 
    mesh that will be used in the network. It then saves the numpy arrays so that at each iteration of the network the loading
    of the data is as efficient as possible.
    """
    dataset.shape_classes = sorted([x.name for x in dataset.dataroot.iterdir() if x.is_dir()])
    
    if not dataset.descriptors:
        new_dataroot = str(dataset.dataroot) + f'_{dataset.nb_augments}_{"_".join(dataset.augments)}_{dataset.mode}_Nodesc'
    else:
        new_dataroot = str(dataset.dataroot) + f'_{dataset.nb_augments}_{"_".join(dataset.augments)}_{dataset.mode}'

    if dataset.reduce:
        new_dataroot = new_dataroot + '_1'
    if dataset.common_edge:
        new_dataroot = new_dataroot+'_common'
    check = os.path.exists(new_dataroot)
    if not check:
        os.mkdir(new_dataroot)
    else: dataset.dataroot= Path(new_dataroot)

    cpt2 = 0

    for obj_class in dataset.dataroot.iterdir():
        if obj_class.is_dir():
            if not check:
                os.mkdir(new_dataroot + '/' + obj_class.name)
                if dataset.split :
                    os.mkdir(new_dataroot + '/' + obj_class.name + '/' + dataset.mode)
            label = dataset.shape_classes.index(obj_class.name)
            if dataset.split :
                dir_iter = (obj_class / dataset.mode).iterdir()
            else:
                dir_iter = (obj_class).iterdir()
            for obj_path in dir_iter:
                if obj_path.is_file():
                    if not check:
                      
                        for i in range(dataset.nb_augments):

                            mesh = trimesh.load(obj_path, process=False)
                            
                            mesh.merge_vertices()
                            mesh.remove_duplicate_faces()
                                
                            for method in dataset.augments:
                                if method == 'orient':
                                    mesh = randomize_mesh_orientation(mesh)
                                if method == 'scale':
                                    mesh = random_scale(mesh)
                                if method == 'lim_dist':
                                    mesh = random_limited_disturbance(mesh)

                            obj_name = str(obj_path.name).split('.')[0]+f'_{i}.npy'

                            if dataset.split :
                                output_path = str(new_dataroot + '/' + obj_class.name + '/' + dataset.mode + '/' + obj_name)
                            else:
                                output_path = str(new_dataroot + '/' + obj_class.name + '/' + obj_name)

                            mesh_features = load_mesh_without_desc(mesh, normalize=True, augments=dataset.augments, request=dataset.feats, K=dataset.K)

                            np.save(output_path, np.array(mesh_features, dtype=object))

                            if dataset.split :
                                dataset.mesh_paths.append(new_dataroot + '/' + obj_class.name + '/' + dataset.mode + '/' + obj_name)
                            else:
                                dataset.mesh_paths.append(new_dataroot + '/' + obj_class.name + '/' + obj_name)
                            dataset.labels.append(label)
                        
                    else:
                        dataset.mesh_paths.append(obj_path)
                        dataset.labels.append(label)

                        name = str(obj_path.name).split('.')[0].split('_')[:-1]
                        name = '_'.join(name)
                        if dataset.mode == 'test' and name not in dataset.files_map:
                            dataset.files_map[name] = cpt2
                            dataset.files.append(cpt2)
                            dataset.file_label.append(label)
                            cpt2+=1
                        elif dataset.mode == 'test':
                            dataset.files.append(dataset.files_map[name])


    dataset.mesh_paths = np.array(dataset.mesh_paths)
    dataset.labels = np.array(dataset.labels)
    dataset.dataroot = new_dataroot