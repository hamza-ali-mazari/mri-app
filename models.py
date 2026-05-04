import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class QMLLayer(nn.Module):
    """Quantum Machine Learning Layer using PennyLane"""
    def __init__(self, n_qubits=8):
        super().__init__()
        self.n_qubits = n_qubits
        # Improved initialization: He initialization for better convergence
        self.weights = nn.Parameter(torch.randn(3, n_qubits) / (3 ** 0.5))
        self.qml_linear = nn.Linear(n_qubits, n_qubits)  # Fallback linear layer
        nn.init.kaiming_normal_(self.qml_linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.qml_linear.bias, 0)
        
        if PENNYLANE_AVAILABLE:
            self.dev = qml.device("default.qubit", wires=n_qubits)
            
            @qml.qnode(self.dev, interface="torch")
            def quantum_circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
            
            self.quantum_circuit = quantum_circuit
    
    def forward(self, x):
        """Forward pass through QML layer
        
        Args:
            x: Input tensor of shape (batch_size, n_qubits)
        
        Returns:
            Output tensor of shape (batch_size, n_qubits)
        """
        if PENNYLANE_AVAILABLE and hasattr(self, 'quantum_circuit'):
            try:
                outputs = []
                for i in range(x.shape[0]):
                    out = self.quantum_circuit(x[i], self.weights)
                    out = torch.stack(out).float()
                    outputs.append(out)
                return torch.stack(outputs)
            except Exception as e:
                # Fallback to linear layer if quantum circuit fails
                return self.qml_linear(x)
        else:
            # Fallback: linear layer if PennyLane not available
            return self.qml_linear(x)

class UNet_QML(nn.Module):
    """U-Net with Quantum Machine Learning components
    
    Input: Grayscale images (1 channel, 64x64)
    Output: Binary segmentation mask (1 channel, 64x64)
    """
    def __init__(self):
        super().__init__()
        
        n_qubits = 8
        
        # Encoder
        self.enc1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)
        self.norm_bottleneck = nn.GroupNorm(16, 256)  # GroupNorm for 4D tensors
        
        # QML components
        self.fc1 = nn.Linear(256, n_qubits)
        self.qml = QMLLayer(n_qubits=n_qubits)
        self.fc2 = nn.Linear(n_qubits, 256)
        self.norm_qml_out = nn.LayerNorm(256)  # LayerNorm for 2D tensors
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        
        # Output
        self.out = nn.Conv2d(32, 1, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))  # (B, 256, 8, 8)
        
        # Improved encoder signal processing with group normalization
        b = self.norm_bottleneck(b)  # Normalize channels (GroupNorm for 4D)
        
        # Global pooling → vector
        b_pool = F.adaptive_avg_pool2d(b, (1, 1)).view(b.shape[0], -1)  # (B, 256)
        b_pool = self.norm_qml_out(b_pool)  # LayerNorm for 2D
        
        # QML pipeline with improved signal flow
        q_in = F.relu(self.fc1(b_pool))      # (B, 8) with ReLU activation
        q_out = self.qml(q_in)               # (B, 8)
        q_out = F.relu(self.fc2(q_out))      # (B, 256) with ReLU activation
        
        # Expand back with careful scaling
        q_out = q_out.unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)
        b = b + 0.1 * q_out  # Smaller residual connection to avoid instability
        
        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)


# Aliases for compatibility
SegmentationModel = UNet_QML
QMLSegmentationModel = UNet_QML
