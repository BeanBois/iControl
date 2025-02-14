# sgl_model.py
import torch
import torch.nn as nn
import numpy as np

class SGLCfg:
    def __init__(self, learning_rate=0.01, tau_pre=20.0, tau_post=20.0):
        self.learning_rate = learning_rate
        self.tau_pre = tau_pre
        self.tau_post = tau_post

class SGLDense(nn.Module):
    """
    Pre-synaptic spikes (x_pre) and post-synaptic spikes (x_post)
    are used to update the weights locally.
    """
    def __init__(self, in_features, out_features, sgl_cfg: SGLCfg):
        super(SGLDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights.
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.sgl_cfg = sgl_cfg
        # For simplicity, we assume that pre- and post-spike activity
        # are available during the forward pass.
    
    def forward(self, x_pre, x_post):
        """
        x_pre: tensor of shape (batch_size, in_features) (input spikes)
        x_post: tensor of shape (batch_size, out_features) (output spikes from previous forward)
        """
        # Compute the output.
        out = torch.matmul(x_pre, self.weight.t())
        # Here, we assume a surrogate STDP update is computed.
        # For demonstration, we use a simple Hebbian update:
        # Δw = learning_rate * (outer(x_post, x_pre))
        # In practice, you would use a surrogate derivative function that incorporates timing.
        delta_w = self.sgl_cfg.learning_rate * torch.matmul(x_post.unsqueeze(2), x_pre.unsqueeze(1)).mean(dim=0)
        # Update weights in a differentiable but local manner.
        # Note: In a real neuromorphic implementation, the weight update would be
        # handled by the process model on the hardware.
        self.weight.data += delta_w
        return out

class SNNControllerSGL(nn.Module):
    """
    A simple spiking neural network controller using SGL.
    It takes a 24-dim observation, processes it through an SGL-enabled dense layer,
    then through a standard linear layer with LIF activation, and outputs a 4-dim action.
    """
    def __init__(self, input_dim=24, hidden_dim=32, output_dim=4):
        super(SNNControllerSGL, self).__init__()
        # First layer: SGL-enabled connection.
        sgl_cfg = SGLCfg(learning_rate=0.005, tau_pre=20.0, tau_post=20.0)
        self.sgl_dense = SGLDense(input_dim, hidden_dim, sgl_cfg)
        # A simple surrogate LIF activation:
        self.lif = lambda x: torch.sigmoid(x)  # surrogate: sigmoid approximates spiking nonlinearity.
        # Second layer: Output mapping.
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        # Final surrogate activation (if desired):
        self.lif_out = lambda x: torch.tanh(x)  # map to -1 to 1
    
    def forward(self, x):
        """
        x: tensor of shape (batch_size, input_dim)
        For SGL, we simulate spikes by thresholding the activations.
        """
        # Generate pre-synaptic spikes from input.
        x_pre = (x > 0.0).float()  # simple thresholding to get spikes.
        # Forward pass through SGL dense layer.
        # Here, we simulate that the post-synaptic spikes are the surrogate activation of the weighted sum.
        pre_activation = torch.matmul(x_pre, self.sgl_dense.weight.t())
        x_post = (pre_activation > 0.5).float()  # simple threshold to generate output spikes.
        hidden = self.sgl_dense(x_pre, x_post)
        hidden = self.lif(hidden)
        # Compute output.
        out = self.fc_out(hidden)
        action = self.lif_out(out)
        return action

# Quick test:
if __name__ == "__main__":
    model = SNNControllerSGL()
    sample_obs = torch.randn(1, 24)
    action = model(sample_obs)
    print("Action from SNN with SGL:", action)
