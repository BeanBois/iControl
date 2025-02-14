# snn_controller.py
import torch
import torch.nn as nn
import lava.dl.slayer as slayer

class SNNController(nn.Module):
    """
    A simple spiking neural network controller built with Lava-DL SLAYER modules.
    It maps a 24-dimensional observation vector to a 4-dimensional action.
    For demonstration purposes, the network consists of two linear layers with LIF activation.
    """
    def __init__(self, input_dim=24, hidden_dim=32, output_dim=4):
        super(SNNController, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # The SLAYER LIF module simulates spiking neuron dynamics.
        self.lif1 = slayer.LIF()  
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = slayer.LIF()
    
    def forward(self, x):
        """
        Forward pass.
        x: tensor of shape (batch_size, input_dim)
        Returns: tensor of shape (batch_size, output_dim)
        """
        out = self.fc1(x)
        out = self.lif1(out)
        out = self.fc2(out)
        out = self.lif2(out)
        return out

# For demonstration, you can test the network with random input:
if __name__ == "__main__":
    model = SNNController()
    sample_obs = torch.randn(1, 24)  # batch_size=1, observation of dimension 24
    action = model(sample_obs)
    print("Sample action:", action)
