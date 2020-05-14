from torch import nn
import torch


class GatLayer(nn.Module):
    def __init__(self, input_feature_dim, output_feature_dim, attention_leakyReLU_slope=0.2):
        super(GatLayer, self).__init__()

        self.weights_matrix = nn.Parameter(torch.empty(
            input_feature_dim, output_feature_dim, dtype=torch.float32))
        self.attention_bias_vector = nn.Parameter(
            torch.empty(2*output_feature_dim, dtype=torch.float32))

        self.__init_parameters()

        self.leaky_relu = nn.LeakyReLU(attention_leakyReLU_slope)

    def __init_parameters(self):
      for parameter in self.parameters():
        #TODO: add gain?
        nn.init.xavier_uniform_(parameter)

    def forward(self, input_matrix, adjacency_coo_matrix):
      

GatLayer(2,3)