import torch
import numpy as np
import torch.nn as nn
import utils

class Mesh_Tensor:
    """
    This class stores the tensors needed to represent a batch of N meshes
    """

    def __init__(self, descriptors = True):
        super(Mesh_Tensor, self).__init__()

        self.faces = torch.Tensor()
        self.edge_features = torch.Tensor()
        self.face_features = torch.Tensor()
        self.adjacency =torch.Tensor()
        self.nb_faces = torch.Tensor()
        self.vertices = torch.Tensor()
        self.ring_k = torch.Tensor()
        self.erased = torch.Tensor()
        self.edges_adjacency = torch.Tensor()
        self.edges = torch.Tensor()
        self.edges_faces = torch.Tensor()
        self.edges_erased = torch.Tensor()
        self.unpool_material = []
        self.enc_reconstuctions = []
        self.dec_reconstuctions = []
        self.latent_faces = None
        self.latent_verts = None
        self.compressed_size = None
        
        self.device = torch.device('cpu')
        self.N = 0
        self.files = torch.Tensor()
    
    def erase_align_edges(self):
        # Décaler edges
        unique,num_to_erase = self.edges_erased[:,0].unique(return_counts=True)
        num_to_erase = num_to_erase.min()

        if unique.size(0) == self.edges.size(0):
            num_edges_to_remove = num_to_erase
            num_edges = self.edges.size(1) - num_edges_to_remove
            to_remove = torch.zeros((self.edges.size(0),num_edges_to_remove)).type(torch.int).to(self.device)
            to_keep = torch.zeros((self.edges.size(0),num_edges)).type(torch.int).to(self.device)
            differences = torch.zeros((self.edges.size(0),self.edges.size(1))).type(torch.int).to(self.device)
            mask_erased = torch.ones(self.edges_erased.shape[0],dtype=bool).to(self.device)
            idx, = torch.where(self.edges_erased[:,1] == self.edges.size(1)-1)
            mask_erased[idx] = False
            
            for i in range(self.edges.size(0)):
                idx0, = torch.where(self.edges_erased[:,0] == i)

                edges_i = self.edges_erased[idx0]
                edges_i = edges_i[:num_to_erase-1]
                mask_erased[idx0[:num_to_erase-1]] = False
                to_remove[i] = torch.cat((edges_i[:,1],torch.ones(1).type(torch.int).to(self.device) * (self.edges.size(1) - 1)))

                mask = torch.ones(self.edges.size(1), dtype=bool)
                mask[to_remove[i]] = False

                to_keep[i] = torch.arange(self.edges.size(1))[mask].type(torch.int).to(self.device)

                mask = torch.zeros(self.edges.size(1), dtype=bool)
                mask[to_remove[i]] = True
                
                differences[i] = torch.cumsum(mask,0)
            differences = torch.cat((differences,torch.zeros(differences.size(0),1).type(torch.int).to(self.device)),dim=1)

            self.edges_erased[mask_erased,1] = self.edges_erased[mask_erased,1] - differences[self.edges_erased[mask_erased][:,0],self.edges_erased[mask_erased][:,1]]
            self.edges_erased = self.edges_erased[mask_erased]
            self.edges_erased = torch.cat((self.edges_erased, torch.cat((torch.arange(self.edges.size(0)).unsqueeze(1).to(self.device), torch.ones(self.edges.size(0)).to(self.device).unsqueeze(1) * num_edges), dim=1)), dim=0).unique(dim=0).type(torch.int)

            self.edges = self.edges[torch.arange(self.edges.size(0))[:,None], to_keep]
            self.edges = torch.cat((self.edges, (torch.ones(self.edges.size(0), 1, self.edges.size(2)) * -1).to(self.device)), dim=1).type(torch.int)
            self.edges_adjacency = self.edges_adjacency - differences[torch.arange(self.edges_adjacency.shape[0])[:,None,None],self.edges_adjacency[torch.arange(self.edges_adjacency.size(0))]]
    
    def edges_faces(self):
        None
    
    def compute_ring_k_batch(self, K):
        """
        This function computes the K ring neighborhood around each face of the meshes in the batch
        """
        num_faces = self.faces.size(1)
        self.ring_k = torch.ones((self.N, num_faces, K)).type(torch.int).to(self.device) * -2
        #initialize the first column with the faces themselves
        self.ring_k[:,:,0] = torch.arange(num_faces)
        #initialize the next three columns with the neighbors of the target faces = the adjacency matrix
        self.ring_k[:,:,1:4] = self.adjacency

        head_ring = torch.ones((self.N, num_faces)).to(self.device).type(torch.int) * 4
        not_erased_0 = torch.arange(self.N).unsqueeze(1).repeat(1,num_faces) 
        not_erased_1 = torch.arange(num_faces).unsqueeze(0).repeat(self.N,1) 
        mask = torch.ones(not_erased_0.numel(),dtype = bool).view(self.N,-1)
        mask[self.erased[:,0],self.erased[:,1]] = False
        not_erased = torch.cat((not_erased_0[mask].unsqueeze(1), not_erased_1[mask].unsqueeze(1)), dim=1).to(self.device)
        not_erased_clone = not_erased.clone().to(self.device).type(torch.int)

        idx0,idx1,idx2 = torch.where(self.adjacency == -1)
        idx2_0, = (torch.where(idx2 == 0))
        self.ring_k[idx0[idx2_0],idx1[idx2_0],1:3] = self.ring_k[idx0[idx2_0],idx1[idx2_0],2:4]
        self.ring_k[idx0[idx2_0],idx1[idx2_0],3] = -1
        idx2_1, = torch.where(idx2 == 1)
        self.ring_k[idx0[idx2_1],idx1[idx2_1],2] = self.ring_k[idx0[idx2_1],idx1[idx2_1],3]
        self.ring_k[idx0[idx2_1],idx1[idx2_1],3] = -1
        unique,counts = torch.cat((idx0.unsqueeze(1),idx1.unsqueeze(1)),dim=1).unique(dim=0,return_counts=True)
        head_ring[unique[:,0],unique[:,1]] = head_ring[unique[:,0],unique[:,1]] - counts.type(torch.int)

        temp = torch.Tensor()
        if K>4:
            # Initialize temp with the neighbors of the three regions taken in order
            temp = self.adjacency[torch.arange(self.N)[:,None,None],self.ring_k[:,:,1:4][None,:,:]].view(self.N,num_faces,-1)
            # Find the faces in temp that have already been added to ring_k (per face and per mesh)
            idx1, idx2, idx3, idx4 = (torch.where(temp.unsqueeze(2) == self.ring_k[:,:,:4].unsqueeze(3)))
            indices = (torch.ones(temp.shape) * torch.arange(temp.shape[2])).type(torch.int)
            # We mark the already added faces with a high number
            indices[idx1,idx2,idx4] = temp.shape[2]+K+1

            idx1, idx2, idx3, idx4 = torch.where(temp.unsqueeze(2) == -1)
            indices[idx1,idx2,idx4] = temp.shape[2]+K+1
            # Select the values that have not yet been added
            indices,_ = torch.min(indices, dim=2)
            idx1, idx2 = torch.where(indices > temp.shape[2])
            indices[idx1,idx2] = 0
            # implement ring_k
            self.ring_k[torch.arange(self.N)[:,None],torch.arange(num_faces)[None,:],head_ring] = temp[torch.arange(self.N)[:,None,None],torch.arange(num_faces)[None,:,None],indices.unsqueeze(2)].view(self.N,num_faces).type(self.ring_k.dtype)
            head_ring += 1
            # repeat the same stages until the desired size is obtained
            # for j in range(5,K): # while (head_ring[not_erased[:,0],not_erased[:,1]] - K).sum() < 0
            while not_erased.size(0)>0:
                temp = torch.cat((temp, self.adjacency[torch.arange(self.N)[:,None],self.ring_k[torch.arange(self.N)[:,None],torch.arange(num_faces)[None,:],head_ring - 1][None,:]].view(self.N,num_faces,-1)), dim=2)
                idx1, idx2, idx3, idx4 = (torch.where(temp.unsqueeze(2) == self.ring_k.unsqueeze(3)))
                idx1, idx2, idx4 = idx1.to(self.device), idx2.to(self.device), idx4.to(self.device)
                indices = (torch.ones(temp.shape) * torch.arange(temp.shape[2])).type(torch.int).to(self.device)
                indices[idx1,idx2,idx4] = temp.shape[2]+K+1 # PROBLEM
                idx1, idx2, idx3, idx4 = (torch.where(temp.unsqueeze(2) == -1))
                indices[idx1,idx2,idx4] = temp.shape[2]+K+1 # PROBLEM

                indices,_ = torch.min(indices, dim=2)
                idx1, idx2 = torch.where(indices > temp.shape[2])
                indices[idx1,idx2] = 0
                self.ring_k[not_erased[:,0],not_erased[:,1],head_ring[not_erased[:,0],not_erased[:,1]]] = temp[not_erased[:,0],not_erased[:,1],indices[not_erased[:,0],not_erased[:,1]]].type(self.ring_k.dtype)#.view(self.N,num_faces)
                head_ring[not_erased[:,0],not_erased[:,1]] += 1
                # print(head_ring[not_erased[:,0],not_erased[:,1]].size())
                idx0 = torch.where(head_ring[not_erased[:,0],not_erased[:,1]] != K)
                not_erased = not_erased[idx0]

    def reconstruct_mesh(self):
        # Mean of everywhere a point is represented in the mesh
        # return inverse?
        max = self.faces.unique().max()
        points_meshes = torch.zeros(self.faces.size(0),max+2,3).to(self.device)

        for i in range(self.faces.size(0)):
            unique, counts = self.faces[i].unique(return_counts = True)
            unique, counts = unique[1:], counts[1:]

            map_vertices = torch.ones(max+2).type(torch.int).to(self.device) * -1
            map_vertices[unique] = torch.arange(unique.size(0)).type(torch.int).to(self.device)

            if self.face_features.size(2) == 9:
                index = self.faces[i].view(-1).unsqueeze(1).repeat(1,3)
                features_resized = self.face_features[i].reshape(self.face_features.size(1)*3,3)
            elif self.face_features.size(2) == 3:
                index = self.faces[i].view(-1).unsqueeze(1).repeat(1,3)
                features_resized = self.face_features[i].repeat(1,3).view(-1,3)
            else:
                print('PROBLEM in the features dimensions')
                return mesh
            index[index==-1] = max+1
            points = torch.zeros(max+2,3).to(self.device).type(features_resized.dtype).scatter_add(0,index,features_resized)
            points[unique] = points[unique]/counts.unsqueeze(1).repeat(1,3)
            points_meshes[i,:unique.size(0)] = points[unique]
            faces = self.faces[i][self.faces[i] != -1].view(-1,3)
            self.faces[i] = map_vertices[self.faces[i]]
            self.edges[i] = map_vertices[self.edges[i]]

        self.vertices = points_meshes[:,:self.faces.unique().max()+2]
        
        return self.vertices, self.faces

    def face_normals(self):
        """
        Calcule les normales des faces d'un mesh triangulaire.
        
        Args:
            vertices (torch.Tensor): (N, 3) positions des sommets.
            faces (torch.Tensor): (M, 3) indices des sommets par face.

        Returns:
            torch.Tensor: (M, 3) normales unitaires de chaque face.
        """
        # Récupérer les sommets de chaque face
        v0 = self.vertices[torch.arange(self.faces.size(0))[:,None], self.faces[:,:, 0]]
        v1 = self.vertices[torch.arange(self.faces.size(0))[:,None], self.faces[:,:, 1]]
        v2 = self.vertices[torch.arange(self.faces.size(0))[:,None], self.faces[:,:, 2]]
        # Calcul des vecteurs d'arêtes
        e1 = v1 - v0
        e2 = v2 - v0
        # Produit vectoriel → normale brute
        normals = torch.cross(e1, e2, dim=2)

        # Normalisation
        normals = normals / (normals.norm(dim=2, keepdim=True) + 1e-8)

        return normals
    
    def compute_valence_per_vertex(self):
        max_vertex = (self.faces.unique().max() + 2).type(torch.long)

        index = self.edges.clone().view(self.faces.size(0),-1).type(torch.long)#self.faces.clone().view(self.faces.size(0),-1).type(torch.long)
        index[index == -1] = max_vertex - 1

        valence_per_vertex = torch.zeros(self.faces.size(0),max_vertex).type(torch.int).to(self.device)
        src = torch.ones(self.faces.size(0),self.edges.size(1) * 2).type(torch.int).to(self.device)
        valence_per_vertex = valence_per_vertex.scatter_add(1,index,src)

        valence_per_vertex[:,max_vertex - 1] = 0
        return valence_per_vertex

    def remove_unused_edges(self):
        edges_erase = torch.arange(self.edges.size(1)).unsqueeze(0).repeat(self.faces.size(0),1).to(self.device)
        edges_erase[torch.arange(self.faces.size(0))[:,None], self.edges_adjacency[:,:,0]] = -1
        edges_erase[torch.arange(self.faces.size(0))[:,None], self.edges_adjacency[:,:,1]] = -1
        edges_erase[torch.arange(self.faces.size(0))[:,None], self.edges_adjacency[:,:,2]] = -1
        edges_erase[:,-1] = 1
        
        idx0,idx1 = torch.where(edges_erase!=-1)
        edges_to_erase = torch.cat((idx0.unsqueeze(1), idx1.unsqueeze(1)),dim=1)

        self.edges_erased = edges_to_erase
        self.erase_align_edges()

        self.edges[self.edges_erased[:,0],self.edges_erased[:,1]] = -1

    def __getitem__(self,i):
        if i == 0: return self.faces
        if i == 1: return self.face_features
        if i == 2: return self.adjacency
        if i == 3: return self.ring_k
        if i == 4 or i == -1: return self.erased
    
    def __setitem__(self,i, new_value):
        if i == 0: self.faces = new_value
        if i == 1: self.face_features = new_value
        if i == 2: self.adjacency = new_value
        if i == 3: self.ring_k = new_value
        if i == 4 or i == -1: self.erased = new_value
    
    def to(self,device):
        self.faces = self.faces.to(device)
        self.edge_features = self.edge_features.to(device)
        self.face_features = self.face_features.to(device)
        self.adjacency = self.adjacency.to(device)
        self.vertices = self.vertices.to(device)
        self.ring_k = self.ring_k.to(device)
        self.erased = self.erased.to(device)
        self.edges_adjacency = self.edges_adjacency.to(device)
        self.edges = self.edges.to(device)
        self.edges_erased = self.edges_erased.to(device)
        self.device = device