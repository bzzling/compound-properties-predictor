import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load and process data
def load_data(file_path):
    data_list = []
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        smiles = row['smiles']
        solubility = row['solubility']
        graph = eval(row['graph'])  # Assuming the graph is stored as a string representation
        graph.y = torch.tensor([solubility], dtype=torch.float)
        graph.edge_index = torch.tensor(graph.edge_index, dtype=torch.long).t().contiguous()
        data_list.append(graph)
    return data_list

# Define the GNN model
class GNNModel(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNNModel, self).__init__()
        self.conv1 = pyg_nn.GraphConv(num_node_features, hidden_channels)
        self.conv2 = pyg_nn.GraphConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = pyg_nn.global_mean_pool(x, batch)
        return self.linear(x)

# Function to train the model
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Function to test the model
def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            total_loss += criterion(out, data.y).item()
    return total_loss / len(loader)

# Main execution
if __name__ == "__main__":
    # Load and process data
    data_list = load_data('data/gnn_input_data.csv')
    
    # Split data into training and testing sets
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Initialize model, optimizer, and loss function
    model = GNNModel(num_node_features=1, hidden_channels=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = test(model, test_loader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')