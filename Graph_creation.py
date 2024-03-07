#import the needed libraries
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import GroupShuffleSplit
import random
import os
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import pickle
from torch import Tensor


def create_groups(in_size, n_groups):
    #having overlapping windows of data, this function aims at creating groups of windows,
    # and determine the index of the window overlapping between two consecutive groups in order to delete it
    remainder_ = in_size % n_groups
    group_size = (in_size - remainder_) / n_groups

    if remainder_ != 0:
        n_groups = n_groups + 1
        # find the index of those epochs
    e_index_to_del = []

    # new_signal = {key: {} for key in input_signal}
    for num in range(1,
                     in_size - remainder_ + 1):  # to delete overlapping epochs between groups without deleting first and last epochs
        if num % group_size == 0:
            # delete last epoch from each group
            e_index_to_del.append(num - 1)
    e_to_keep = [x for x in np.arange(in_size) if x not in e_index_to_del]
    numbers = np.repeat(np.arange(1, n_groups), group_size - 1)
    groups = np.concatenate((numbers, np.repeat(n_groups, remainder_)))

    return e_index_to_del, e_to_keep, groups

def get_edge_attributes(weight_matrix):
    # Get the shape of the adjacency matrix to determine the number of nodes
    num_nodes = weight_matrix.shape[0]
    # Create a list to store the edge attributes (weights)
    edge_attributes_list = []
    # Loop through the adjacency matrix to extract edge attributes
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and weight_matrix[i, j] != 0:
                # Avoid self-loops by checking i != j
                edge_attributes_list.append(weight_matrix[i, j])
    # Convert the list of edge attributes into a PyTorch tensor
    edge_attributes_tensor = torch.tensor(edge_attributes_list, dtype=torch.float32)
    return edge_attributes_tensor

def normalized(features,norm_type='linear scaling'):
    if norm_type == 'z_scores':
        mean, _ = torch.mean(features,dim=0)
        std , _ = torch.std(features,dim=0)
        return torch.div((features - mean),std)
    elif norm_type == 'linear scaling':
        max_e, _ = torch.max(features,dim=0)
        min_e , _ = torch.min(features,dim=0)
        return torch.div((features - min_e),(max_e - min_e))

def transform_to_consecutive(numbers):
    result = [1]
    current_number = numbers[0]
    consecutive_count = 1
    for num in numbers[1:]:
        if num != current_number:
            current_number = num
            consecutive_count += 1
            result.append(consecutive_count)
        else:
            # consecutive_count += 1
            result.append(consecutive_count)
        # result.append(consecutive_count)
    return result

participant = 2
session = 1
class_names = ['easy','med','diff']
n_groups = 27 #in our case we use 27 groups to have groups of the same size
batch_size = 60 #try your own values
# To load energy features
graph_path = f'yourpath/graphs'
val_size =0.1 #take 10% from train to validate
fold = 0 #if you want 5 folds cross_validation  iterate for fold in range(5)
train_size = 0.8
test_size = 0.2

energy_features_path = f'yourpath/relative_psd_features_S{session}.pkl'
with open(energy_features_path, 'rb') as pickle_file:
    all_energy = pickle.load(pickle_file)
label = 0
data_list = [] #will store all the graph objects
print('creating graphs')


# code that creates graphs with nodes labels and weights of non-filtered signals as edge features
for class_name in class_names:
    # load weigth matrix and create groups
    raw_file_name = f'Weight_{class_name}_S{session}_no_filters.npy'
    weight_matrices = torch.tensor(np.load(os.path.join(graph_path, raw_file_name)))
    e_index_to_del, e_index_to_keep, groups = create_groups(len(weight_matrices), n_groups) #epochs to remove while creating groups to avoid overlapping epochs

    for i in e_index_to_keep:
        edge_index = torch.nonzero(weight_matrices[i])  # Edge connections
        edge_attributes = get_edge_attributes(weight_matrices[i])
        # Create a `Data` object for this graph.
        e_features = torch.tensor(all_energy[class_name][i], dtype=torch.float32) # remove energies from the deleted epochs

        # normalize the features per node
        norm_efeatures = normalized(e_features, 'linear scaling')
        #having the features, we create the graph object, labels will be 0->low, 1->med, 2->diff
        gr_data = Data(x=norm_efeatures, edge_index=edge_index.t(),
                       edge_attr=edge_attributes, y=torch.tensor(label))
        data_list.append(gr_data)
    label += 1

# now having the indices of train/test splits to use, we will load them directly and create our dataloaders
# this dictionary contains splits for 5 folds (could be used for cross validation)
with open(
        f'yourpath/split27grps_{train_size}_train.pickle',
        'rb') as file:
    train_test_splits = pickle.load(file)

data_train_val = []
data_test = []
data_val = []
data_train = []

class_len = int(len(data_list) / len(class_names))
for i in range(len(class_names)):
    data_train_val.extend(data_list[j] for j in train_test_splits['X_train'][fold]['index'] + i * class_len)
    data_test.extend(data_list[j] for j in train_test_splits['X_test'][fold]['index'] + i * class_len)

train_val_class_len = int(len(data_train_val) / len(class_names))
gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random.randint(40, 60))
train_index, val_index = next(
    gss.split(data_train_val[0:int(len(data_train_val) / 3)], np.ones(int(len(data_train_val) / 3)),
              groups=np.squeeze(transform_to_consecutive(train_test_splits['X_train'][fold]['groups']))))
for i in range(len(class_names)):
    data_train.extend(data_train_val[j] for j in train_index + i * train_val_class_len)
    data_val.extend(data_train_val[j] for j in val_index + i * train_val_class_len)
train_loader = DataLoader(data_train, batch_size=batch_size)
test_loader = DataLoader(data_test, batch_size=len(data_test))  # treat all graphs together
val_loader = DataLoader(data_val, batch_size=batch_size)

