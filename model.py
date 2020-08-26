import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLDL_model(nn.Module):

    def __init__(self, args):

        super().__init__()
        self.args = args
        self.NetworkStructure = args['NetworkStructure']
        self.name_list = ['Layer 0 ({})'.format(self.NetworkStructure[0])]
        self.index_list = [0]
        self.network = nn.ModuleList()

        # Encoder
        for i in range(len(self.NetworkStructure)-1):
            self.network.append(
                nn.Linear(
                    self.NetworkStructure[i], self.NetworkStructure[i+1])
            )

            self.name_list.append('Layer {} ({})'.format(i+1, self.NetworkStructure[i+1]))

            if i != len(self.NetworkStructure)-2:
                if 'Spheres' in self.args['DATASET']:
                    self.network.append(nn.LeakyReLU())
                else:
                    self.network.append(nn.LeakyReLU(0.1))
                self.name_list.append('Layer {} ({})'.format(i+1, self.NetworkStructure[i+1]))

            self.index_list.append(len(self.name_list)-1)
            
        # Decoder
        for i in range(len(self.NetworkStructure)-1, 0, -1):
            self.network.append(
                nn.Linear(
                    self.NetworkStructure[i], self.NetworkStructure[i-1])
            )

            self.name_list.append('Layer {}\' ({})'.format(i-1, self.NetworkStructure[i-1]))

            if i > 1:
                if 'Spheres' in self.args['DATASET']:
                    self.network.append(nn.LeakyReLU())
                else:
                    self.network.append(nn.LeakyReLU(0.1))
                self.name_list.append('Layer {}\' ({})'.format(i-1, self.NetworkStructure[i-1]))

            self.index_list.append(len(self.name_list)-1)
    
    # Forward, and saves all intermediate results as a list
    def forward(self, data):

        data = data.view(data.shape[0], -1)
        output_info = [data, ]
        input_data = data

        for i, layer in enumerate(self.network):
            output_data = layer(input_data)
            output_info.append(output_data)
            input_data = output_data

        return output_info

    # Input the input layer data, pass the encoder, and get the reconstruction result
    def Encoder(self, data):

        output_info = []
        input_data = data

        for i, layer in enumerate(self.network):
            if i <= (len(self.NetworkStructure) - 2) * 2:
                output_data = layer(input_data)
                output_info.append(output_data)
                input_data = output_data

        return output_info[-1]

    # Input the hidden layer data, pass the decoder, and get the reconstruction result
    def Decoder(self, data):

        output_info = []
        input_data = data

        for i, layer in enumerate(self.network):
            if i > (len(self.NetworkStructure) - 2) * 2:
                output_data = layer(input_data)
                output_info.append(output_data)
                input_data = output_data

        return output_info[-1]