import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, zdims, n_frames):
        super().__init__()
        self.zdims = zdims
        self.n_frames = n_frames

        # --- Encoder Layers ---
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.Tanh(),
            nn.Conv1d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Conv1d(8, 12, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(12),
            nn.Tanh(),
            nn.Conv1d(12, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.Tanh()
        )

        # --- Decoder Layers (defined individually) ---
        self.deconv1 = nn.ConvTranspose1d(16, 12, kernel_size=4, stride=2, padding=1)
        self.bn_d1 = nn.BatchNorm1d(12)
        self.act_d1 = nn.Tanh()
        self.deconv2 = nn.ConvTranspose1d(12, 8, kernel_size=4, stride=2, padding=1)
        self.bn_d2 = nn.BatchNorm1d(8)
        self.act_d2 = nn.Tanh()
        self.deconv3 = nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1)
        self.bn_d3 = nn.BatchNorm1d(4)
        self.act_d3 = nn.Tanh()
        self.deconv4 = nn.ConvTranspose1d(4, 1, kernel_size=4, stride=2, padding=1)
        self.act_d4 = nn.Tanh()

        self._determine_fc_size(self.n_frames)

    def _determine_fc_size(self, n_frames_sample):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.n_frames)
            encoder_output = self.encoder(dummy_input)
            flattened_size = encoder_output.flatten(1).shape[1]
        self.fc_e = nn.Linear(flattened_size, self.zdims)
        self.fc_d = nn.Linear(self.zdims, flattened_size)
        self.encoder_output_shape = encoder_output.shape

    # THIS HELPER FUNCTION IS THE "ESTIMATION" METHOD
    def _generate_default_sizes(self, target_output_len):
        sizes = []
        x_in = torch.zeros(1, 1, target_output_len)
        for layer in self.encoder:
            if isinstance(layer, nn.Conv1d):
                sizes.append(x_in.size(2))
            x_in = layer(x_in)
        return sizes

    def encode(self, x):
        input_sizes = []
        x_in = x
        for layer in self.encoder:
            if isinstance(layer, nn.Conv1d):
                input_sizes.append(x_in.size(2))
            x_in = layer(x_in)
        output = x_in.flatten(1)
        z = self.fc_e(output)
        return z, input_sizes

    # `input_sizes=None` makes the argument optional
    def decode(self, z, input_sizes=None):
        # This `if` block handles the case when `decode` is called without `input_sizes`
        if input_sizes is None:
            # This is how we "estimate" the input sizes: by generating a default
            # list based on the target `n_frames` the model was created with.
            input_sizes = self._generate_default_sizes(self.n_frames)

        x = self.fc_d(z)
        x = x.view(-1, self.encoder_output_shape[1], self.encoder_output_shape[2])

        # The rest of the decode method proceeds as before, using the generated list
        target_size = input_sizes.pop()
        op = self._calculate_output_padding(x.size(2), target_size, self.deconv1.kernel_size[0], self.deconv1.stride[0], self.deconv1.padding[0])
        x = F.conv_transpose1d(x, self.deconv1.weight, self.deconv1.bias, stride=self.deconv1.stride, padding=self.deconv1.padding, output_padding=op)
        x = self.bn_d1(x)
        x = self.act_d1(x)

        target_size = input_sizes.pop()
        op = self._calculate_output_padding(x.size(2), target_size, self.deconv2.kernel_size[0], self.deconv2.stride[0], self.deconv2.padding[0])
        x = F.conv_transpose1d(x, self.deconv2.weight, self.deconv2.bias, stride=self.deconv2.stride, padding=self.deconv2.padding, output_padding=op)
        x = self.bn_d2(x)
        x = self.act_d2(x)

        target_size = input_sizes.pop()
        op = self._calculate_output_padding(x.size(2), target_size, self.deconv3.kernel_size[0], self.deconv3.stride[0], self.deconv3.padding[0])
        x = F.conv_transpose1d(x, self.deconv3.weight, self.deconv3.bias, stride=self.deconv3.stride, padding=self.deconv3.padding, output_padding=op)
        x = self.bn_d3(x)
        x = self.act_d3(x)

        target_size = input_sizes.pop()
        op = self._calculate_output_padding(x.size(2), target_size, self.deconv4.kernel_size[0], self.deconv4.stride[0], self.deconv4.padding[0])
        x = F.conv_transpose1d(x, self.deconv4.weight, self.deconv4.bias, stride=self.deconv4.stride, padding=self.deconv4.padding, output_padding=op)
        reconstructed = self.act_d4(x)
      
        return reconstructed
    
    def _calculate_output_padding(self, input_size, output_size, kernel_size, stride, padding):
        return output_size - ((input_size - 1) * stride - 2 * padding + kernel_size)

    def forward(self, x):
        z, sizes = self.encode(x)
        reconstructed = self.decode(z, sizes)
        return reconstructed