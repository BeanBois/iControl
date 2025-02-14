# snn_controller_stdp.py
import torch
import torch.nn as nn
import numpy as np

class STDPCfg:
    def __init__(self, learning_rate=0.01, tau_pre=20.0, tau_post=20.0):
        self.learning_rate = learning_rate
        self.tau_pre = tau_pre
        self.tau_post = tau_post

class STDPDense(nn.Module):
    """
    A fully connected layer with a simple STDP rule.
    Pre-synaptic spikes (x_pre) and post-synaptic spikes (x_post)
    are used to update the weights locally.
    """
    def __init__(self, in_features, out_features, stdp_cfg: STDPCfg):
        super(STDPDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights.
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.stdp_cfg = stdp_cfg
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
        delta_w = self.stdp_cfg.learning_rate * torch.matmul(x_post.unsqueeze(2), x_pre.unsqueeze(1)).mean(dim=0)
        # Update weights in a differentiable but local manner.
        # Note: In a real neuromorphic implementation, the weight update would be
        # handled by the process model on the hardware.
        self.weight.data += delta_w
        return out

class SNNControllerSTDP(nn.Module):
    """
    A simple spiking neural network controller using STDP.
    It takes a 24-dim observation, processes it through an STDP-enabled dense layer,
    then through a standard linear layer with LIF activation, and outputs a 4-dim action.
    """
    def __init__(self, input_dim=24, hidden_dim=32, output_dim=4):
        super(SNNControllerSTDP, self).__init__()
        # First layer: STDP-enabled connection.
        stdp_cfg = STDPCfg(learning_rate=0.005, tau_pre=20.0, tau_post=20.0)
        self.stdp_dense = STDPDense(input_dim, hidden_dim, stdp_cfg)
        # A simple surrogate LIF activation:
        self.lif = lambda x: torch.sigmoid(x)  # surrogate: sigmoid approximates spiking nonlinearity.
        # Second layer: Output mapping.
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        # Final surrogate activation (if desired):
        self.lif_out = lambda x: torch.tanh(x)  # map to -1 to 1
    
    def forward(self, x):
        """
        x: tensor of shape (batch_size, input_dim)
        For STDP, we simulate spikes by thresholding the activations.
        """
        # Generate pre-synaptic spikes from input.
        x_pre = (x > 0.0).float()  # simple thresholding to get spikes.
        # Forward pass through STDP dense layer.
        # Here, we simulate that the post-synaptic spikes are the surrogate activation of the weighted sum.
        pre_activation = torch.matmul(x_pre, self.stdp_dense.weight.t())
        x_post = (pre_activation > 0.5).float()  # simple threshold to generate output spikes.
        hidden = self.stdp_dense(x_pre, x_post)
        hidden = self.lif(hidden)
        # Compute output.
        out = self.fc_out(hidden)
        action = self.lif_out(out)
        return action

# Quick test:
if __name__ == "__main__":
    model = SNNControllerSTDP()
    sample_obs = torch.randn(1, 24)
    action = model(sample_obs)
    print("Action from SNN with STDP:", action)
