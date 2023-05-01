from io import BytesIO
import pyarrow as pa
import pickle

import numpy as np
import pandas as pd

import json
import collections
import math
import os
import collections
import itertools
from itertools import combinations
from shutil import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch.autograd import Variable
from torch_geometric.nn import GCN
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as DataLoaderGraph
from torch_geometric.data import Dataset as DatasetGraph
from torch_geometric.data import Batch as BatchGraph

# For Link Prediction
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero



from typing import Union

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        x = self.convs[self.num_layers](x, edge_index)
        return x

# Can be used to understand above code:
# class SimpleGNN(nn.Module):
#     def __init__(self,num_of_feat, hidden_layer, num_class):
#         super(SimpleGNN, self).__init__()
#         self.conv1 = GCNConv(num_of_feat, hidden_layer)
#         self.conv2 = GCNConv(hidden_layer, num_class)

#     def forward(self, data):
#         x = data.x.float()
#         edge_index = data.edge_index
#         x = self.conv1(x=x, edge_index=edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x

class AddMask(object):
    """
    Recive a graph object and a table.
    Create a mask with only table rows.
    """
    def __init__(self, key):
        self.key = key

    def __call__(self,graph,data):
        data_mask = self.__data_mask__(graph, data)
        graph.data_mask = data_mask
        return graph
    
    def __data_mask__(self, graph, data):
        data_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        mask_idx = data[self.key].to_list()
        data_mask[mask_idx]=True
        return data_mask 
    
    def __repr__(self):
        return '{}(split={})'.format(self.__class__.__name__, self.split)


########## LINK PREDICTION #############

class Link_Prediction_GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user, x_movie, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = Link_Prediction_GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData):
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        } 

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )

        return pred
