import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a tiny dataset: 2 samples, 2 features each
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([0, 1])
batch_size = 2

# Define a very simple neural network
class TinyNN(nn.Module):
    def __init__(self):
        super(TinyNN, self).__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)

# Function to print model parameters and gradients
def print_model_info(model, label):
    print(f'\n{label}')
    for name, param in model.named_parameters():
        print(f'{name} value:\n{param.data}')
        if param.grad is not None:
            print(f'{name} grad:\n{param.grad}')
        else:
            print(f'{name} grad: None')

# Batch loss calculation and backpropagation
def batch_backward(model, X, y, criterion):
    model.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    print(f'Batch Loss: {loss.item():.4f}')
    print_model_info(model, 'After Batch Backward')

# Individual loss calculation and backpropagation with gradient accumulation
def individual_backward(model, X, y, criterion):
    model.zero_grad()
    total_loss = 0.0
    for i in range(X.size(0)):
        single_X = X[i:i+1]
        single_y = y[i:i+1]
        output = model(single_X)
        loss = criterion(output, single_y)
        loss.backward()  # Gradients accumulate by default
        total_loss += loss.item()
    avg_loss = total_loss / X.size(0)
    print(f'Individual Average Loss: {avg_loss:.4f}')
    print_model_info(model, 'After Individual Backward')

# Main execution
def main():
    # Initialize two identical models for comparison
    model_batch = TinyNN()
    model_individual = TinyNN()
    
    # Copy weights to ensure they start identical
    model_individual.load_state_dict(model_batch.state_dict())
    
    criterion = nn.CrossEntropyLoss()
    
    print_model_info(model_batch, 'Initial Model (Batch)')
    print_model_info(model_individual, 'Initial Model (Individual)')
    
    # Perform batch backward
    print('\n=== Batch Backward ===')
    batch_backward(model_batch, X, y, criterion)
    
    # Perform individual backward
    print('\n=== Individual Backward ===')
    individual_backward(model_individual, X, y, criterion)

if __name__ == '__main__':
    main() 