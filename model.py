import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

def test(model, loader):
    model.eval()
    total_loss = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            total_loss += criterion(out, data.y).item()
    return total_loss / len(loader)

class GCNPungencyPredictor(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCNPungencyPredictor, self).__init__()
        self.conv1 = pyg_nn.GCNConv(num_node_features, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = pyg_nn.global_mean_pool(x, batch)
        return self.linear(x)

model = GCNPungencyPredictor(num_node_features=1, hidden_channels=64)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

def train(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# Assuming you've loaded your data into a list called 'dataset'
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(100):
    train(model, loader, optimizer, criterion)

test_loader = DataLoader(test_dataset, batch_size=32)
test_loss = test(model, test_loader)
print(f"Test Loss: {test_loss}")
