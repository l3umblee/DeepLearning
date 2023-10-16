import torch
import torch.nn.functional as F
# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Reproducibility
torch.manual_seed(123)
if device == 'cuda':
 torch.cuda.manual_seed_all(123)
# Input
X = torch.FloatTensor([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]).to(device)
Y = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]).to(device)
# Model
layer = torch.nn.Sequential(
 torch.nn.Linear(in_features=2, out_features=10, bias=True),
 torch.nn.Linear(in_features=10, out_features=3, bias=True),
 torch.nn.ReLU(),
).to(device)
# Setup optimizer
optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

# Training
for step in range(100):
    # Extract output of single layer
    hypothesis = layer(X)
    # Calculate cost
    cost = (-Y.detach() * F.log_softmax(hypothesis, dim=1)).sum(dim=1).mean()
    # Gradient initialization
    optimizer.zero_grad()
    # Calculate gradient
    cost.backward()
    # Update parameters
    optimizer.step()
    if step % 20 == 0:
        print(f'{step}: Loss: {cost.item()}')

print('-------------------------')
for iter, test_input in enumerate(X):
    with torch.no_grad():
        hypothesis = layer(test_input)
    # Softmax
    prob = hypothesis.softmax(dim=-1)
    # Prediction
    max_prob, max_idx = prob.max(dim=-1)
    # One-hot encoding
    pred = torch.eye(n=3)[max_idx].to(device)
    # Accuracy
    print(
    f'Iteration: {iter+1},'
    f' Input: {test_input.detach().cpu().numpy()},'
    f' output: {pred.detach().cpu().numpy()},'
    f' ground-truth: {Y[iter].detach().cpu().numpy()}')