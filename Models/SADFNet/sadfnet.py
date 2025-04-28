import torch
import torch.nn as nn
from .convolutional_layers import ConvolutionalFeatureExtractor
from .recurrent_layers import TemporalModel
from .attention_module import AttentionFusion

class SADFNet(nn.Module):
    def __init__(self, image_channels, sensor_input_size, hidden_size, attention_dim, output_size):
        super(SADFNet, self).__init__()
        self.spatial_extractor = ConvolutionalFeatureExtractor(image_channels, 128)
        self.temporal_model = TemporalModel(sensor_input_size, hidden_size)
        self.attention_fusion = AttentionFusion(hidden_size, attention_dim)
        self.fc = nn.Linear(attention_dim, output_size)

    def forward(self, image_data, sensor_data):
        spatial_features = self.spatial_extractor(image_data)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)

        temporal_output, _ = self.temporal_model(sensor_data)
        last_hidden = temporal_output[:, -1, :]

        combined_features = torch.cat([spatial_features, last_hidden], dim=1)
        fused_features = self.attention_fusion(combined_features.unsqueeze(1)).squeeze(1)

        output = self.fc(fused_features)
        return output
