import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d as BatchNorm
from torch_geometric.nn import GATConv, GraphNorm, global_mean_pool

def define_GNN_structure(num_layers, hidden_channels, feature_dim, target_dim):
    
    class StructureError(Exception):
        pass
    
    class GNN(torch.nn.Module):
        def __init__(self):
            super(GNN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GATConv(feature_dim, hidden_channels)
            self.conv1_norm = GraphNorm(hidden_channels)
            
            if num_layers > 1:
                self.conv2 = GATConv(hidden_channels, hidden_channels)
                self.conv2_norm = GraphNorm(hidden_channels)
            if num_layers > 2:
                self.conv3 = GATConv(hidden_channels, hidden_channels)
                self.conv3_norm = GraphNorm(hidden_channels)
            if num_layers > 3:
                self.conv4 = GATConv(hidden_channels, hidden_channels)
                self.conv4_norm = GraphNorm(hidden_channels)
            if num_layers > 4:
                self.conv5 = GATConv(hidden_channels, hidden_channels)
                self.conv5_norm = GraphNorm(hidden_channels)
            if num_layers > 5:
                self.conv6 = GATConv(hidden_channels, hidden_channels)
                self.conv6_norm = GraphNorm(hidden_channels)
            if num_layers > 6:
                self.conv7 = GATConv(hidden_channels, hidden_channels)
                self.conv7_norm = GraphNorm(hidden_channels)
            if num_layers > 7:
                self.conv8 = GATConv(hidden_channels, hidden_channels)
                self.conv8_norm = GraphNorm(hidden_channels)
            if num_layers > 8:
                raise StructureError("GNN structure only supports up to 8 layers")
            
            self.lin1 = Linear(hidden_channels, hidden_channels)
            self.lin1_norm = BatchNorm(hidden_channels)
            self.lin2 = Linear(hidden_channels, 128)
            self.lin2_norm = BatchNorm(128)
            self.lin3 = Linear(128, target_dim)

        def forward(self, x, edge_index, edge_attr, batch):      
            x = self.conv1(x, edge_index, edge_attr)
            x = self.conv1_norm(x)
            x = F.leaky_relu(x)
            if num_layers > 1:
                x = self.conv2(x, edge_index, edge_attr)
                x = self.conv2_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 2:
                x = self.conv3(x, edge_index, edge_attr)
                x = self.conv3_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 3:
                x = self.conv4(x, edge_index, edge_attr)
                x = self.conv4_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 4:
                x = self.conv5(x, edge_index, edge_attr)
                x = self.conv5_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 5:
                x = self.conv6(x, edge_index, edge_attr)
                x = self.conv6_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 6:
                x = self.conv7(x, edge_index, edge_attr)
                x = self.conv7_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 7:
                x = self.conv8(x, edge_index, edge_attr)
                x = self.conv8_norm(x)
                x = F.leaky_relu(x)

            #Returns batch-wise graph-level-outputs by averaging node features across the node dimension, so that for a single graph G
            #its output is computed by
            x = global_mean_pool(x, batch) 
            
            x = self.lin1(x)
            x = self.lin1_norm(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            
            x = self.lin2(x)
            x = self.lin2_norm(x)
            x = F.leaky_relu(x)
            
            x = self.lin3(x)
            
            return x 
    
    return GNN # returns a class, not an instance of a class. Create model by calling GNN()


def define_GNN_structure_with_global_features(num_layers, hidden_channels, node_feature_dim, features_dim, target_dim):
    
    class StructureError(Exception):
        pass
    
    class GNN(torch.nn.Module):
        def __init__(self):
            super(GNN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GATConv(node_feature_dim, hidden_channels)
            self.conv1_norm = GraphNorm(hidden_channels)
            
            if num_layers > 1:
                self.conv2 = GATConv(hidden_channels, hidden_channels)
                self.conv2_norm = GraphNorm(hidden_channels)
            if num_layers > 2:
                self.conv3 = GATConv(hidden_channels, hidden_channels)
                self.conv3_norm = GraphNorm(hidden_channels)
            if num_layers > 3:
                self.conv4 = GATConv(hidden_channels, hidden_channels)
                self.conv4_norm = GraphNorm(hidden_channels)
            if num_layers > 4:
                self.conv5 = GATConv(hidden_channels, hidden_channels)
                self.conv5_norm = GraphNorm(hidden_channels)
            if num_layers > 5:
                self.conv6 = GATConv(hidden_channels, hidden_channels)
                self.conv6_norm = GraphNorm(hidden_channels)
            if num_layers > 6:
                self.conv7 = GATConv(hidden_channels, hidden_channels)
                self.conv7_norm = GraphNorm(hidden_channels)
            if num_layers > 7:
                self.conv8 = GATConv(hidden_channels, hidden_channels)
                self.conv8_norm = GraphNorm(hidden_channels)
            if num_layers > 8:
                raise StructureError("GNN structure only supports up to 8 layers")
            
            self.input_norm = BatchNorm(hidden_channels+features_dim)
            self.lin1 = Linear(hidden_channels+features_dim, hidden_channels)
            self.lin2 = Linear(hidden_channels, 512)
            self.lin3 = Linear(512, target_dim)

        def forward(self, x, edge_index, edge_attr, batch, features):      
            x = self.conv1(x, edge_index, edge_attr)
            x = self.conv1_norm(x)
            x = F.leaky_relu(x)
            if num_layers > 1:
                x = self.conv2(x, edge_index, edge_attr)
                x = self.conv2_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 2:
                x = self.conv3(x, edge_index, edge_attr)
                x = self.conv3_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 3:
                x = self.conv4(x, edge_index, edge_attr)
                x = self.conv4_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 4:
                x = self.conv5(x, edge_index, edge_attr)
                x = self.conv5_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 5:
                x = self.conv6(x, edge_index, edge_attr)
                x = self.conv6_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 6:
                x = self.conv7(x, edge_index, edge_attr)
                x = self.conv7_norm(x)
                x = F.leaky_relu(x)
            if num_layers > 7:
                x = self.conv8(x, edge_index, edge_attr)
                x = self.conv8_norm(x)
                x = F.leaky_relu(x)

            #Returns batch-wise graph-level-outputs by averaging node features across the node dimension, so that for a single graph G
            #its output is computed by
            x = global_mean_pool(x, batch)
            x2 = features
            x = torch.cat((x, x2), dim=1) 
            
            x = self.input_norm(x)
            x = self.lin1(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            
            x = self.lin2(x)
            x = F.leaky_relu(x)
            
            x = self.lin3(x)
            
            return x 
    
    return GNN # returns a class, not an instance of a class. Create model by calling GNN()
