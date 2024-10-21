import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.solubility)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    total_error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_error += F.mse_loss(output, data.solubility).item()
    return total_error / len(loader)

def main():
    # Load data
    data_list = torch.load('data/processed_data.pt')
    
    # Add solubility as a graph-level property
    for data in data_list:
        data.solubility = data['log solubility (M)']
    
    # Split data into training and test sets
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Initialize model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Train the model
    for epoch in range(100):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_error = test(model, test_loader, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Error: {test_error:.4f}')

if __name__ == "__main__":
    main()