import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

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
        return x.squeeze()  # Ensure the output shape is [batch_size]

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
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_error += F.mse_loss(output, data.solubility).item()
            all_outputs.append(output.cpu().numpy())
            all_targets.append(data.solubility.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    mae = mean_absolute_error(all_targets, all_outputs)
    r2 = r2_score(all_targets, all_outputs)
    
    return total_error / len(loader), mae, r2

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = torch.nn.MSELoss()
    
    best_r2 = float('-inf')
    patience = 20
    patience_counter = 0
    
    # Train the model
    for epoch in range(100):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_error, mae, r2 = test(model, test_loader, device)
        scheduler.step(test_error)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Error: {test_error:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
        
        # Early stopping
        if r2 > best_r2:
            best_r2 = r2
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    main()