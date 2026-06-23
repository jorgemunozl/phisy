import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 1. Create the grid (the "knots" for the spline)
        # We extend the grid range slightly to handle edge cases
        h = (1 - (-1)) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + (-1)).expand(in_features, -1)
        self.register_buffer("grid", grid)

        # 2. Learnable weights (coefficients) for the splines
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        # 3. Learnable "Base" weight (like a standard linear layer shortcut)
        # KANs often add this silicon-based "residual" for stability
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    def b_splines(self, x):
        """
        Compute B-spline bases for input x using Cox-de Boor recursion.
        This expands 1 input scalar into 'grid_size + spline_order' basis values.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid  # (in_features, grid_points)
        x = x.unsqueeze(-1) # (batch, in, 1)
        
        # Base case: Degree 0 (rectangular pulses)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
        
        # Recursion: Degree k depends on Degree k-1
        for k in range(1, self.spline_order + 1):
            # Term 1: (x - t_i) / (t_{i+k} - t_i)
            t_i = grid[:, :-(k+1)]
            t_ik = grid[:, k:-1]
            term1 = (x - t_i) / (t_ik - t_i) * bases[:, :, :-1]
            
            # Term 2: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1})
            t_ip1 = grid[:, 1:-k]
            t_ikp1 = grid[:, k+1:]
            term2 = (t_ikp1 - x) / (t_ikp1 - t_ip1) * bases[:, :, 1:]
            
            bases = term1 + term2
            
        return bases # Shape: (batch, in, basis_count)

    def forward(self, x):
        # Shape: (batch, in_features)
        
        # 1. Compute Spline Output
        # Expand x into basis functions
        bases = self.b_splines(x) # (batch, in, coeff_dim)
        # Multiply by learnable weights
        spline_output = F.linear(bases.view(x.size(0), -1), self.spline_weight.view(self.out_features, -1))
        
        # 2. Compute Base (Linear) Output
        base_output = F.linear(F.silu(x), self.base_weight)
        
        return base_output + spline_output

# --- Testing the Script ---
import math

# Define a simple KAN network (1 input -> 8 hidden -> 1 output)
model = nn.Sequential(
    KANLayer(1, 8),
    KANLayer(8, 1)
)

# Create data: y = sin(3x)
x_train = torch.linspace(-1, 1, 100).unsqueeze(1)
y_train = torch.sin(3 * x_train)

optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
print("Training KAN from scratch...")
for epoch in range(500):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = F.mse_loss(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

plt.plot(x_train.numpy(), y_train.numpy(), label='True')
plt.plot(x_train.numpy(), y_pred.detach().numpy(), label='Predicted')
plt.legend()
plt.show()

print("Done! Final Loss:", loss.item())