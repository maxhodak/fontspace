from torch import nn

LAYERS = 3
KERNELS = [3, 3, 3]
CHANNELS = [32, 64, 128]
STRIDES = [2, 2, 2]
LINEAR_DIM = 8192


class Encoder(nn.Module):
    def __init__(self, output_dim=2, use_batchnorm=False, use_dropout=False):
        super(Encoder, self).__init__()

        self.output_dim = output_dim
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS
        self.strides = STRIDES
        self.conv = self.get_convs()

        self.fc_dim = LINEAR_DIM
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.fc_dim, self.output_dim)

    def get_convs(self):
        conv_layers = nn.Sequential()
        conv_layers.append(
                    nn.Conv2d(
                        1,
                        self.channels[0],
                        kernel_size=self.kernels[0],
                        stride=self.strides[0],
                        padding=1))
        if self.use_batchnorm:
            conv_layers.append(nn.BatchNorm2d(self.channels[0]))
        conv_layers.append(nn.GELU())
        if self.use_dropout:
            conv_layers.append(nn.Dropout2d(0.15))
        for i in range(1, self.layers):
            conv_layers.append(
                nn.Conv2d(
                    self.channels[i - 1],
                    self.channels[i],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=1))
            if self.use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(self.channels[i]))
            conv_layers.append(nn.GELU())
            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))

        return conv_layers

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim=2, use_batchnorm=False, use_dropout=False):
        super(Decoder, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.fc_dim = LINEAR_DIM
        self.input_dim = input_dim
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS[::-1]  # flip the channel dimensions
        self.strides = STRIDES
        self.linear = nn.Linear(self.input_dim, self.fc_dim)
        self.conv = self.get_convs()
        self.output = nn.Conv2d(self.channels[-1], 1, kernel_size=1, stride=1)

    def get_convs(self):
        conv_layers = nn.Sequential()
        conv_layers.append(
                    nn.ConvTranspose2d(
                        self.channels[0],
                        self.channels[0],
                        kernel_size=self.kernels[0],
                        stride=self.strides[0],
                        padding=1,
                        output_padding=1))
        
        if self.use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(self.channels[0]))

        conv_layers.append(nn.GELU())

        if self.use_dropout:
            conv_layers.append(nn.Dropout2d(0.15))
                
        for i in range(1, self.layers):
            conv_layers.append(
                nn.ConvTranspose2d(
                    self.channels[i - 1],
                    self.channels[i],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=1,
                    output_padding=1,
                )
            )

            if self.use_batchnorm and i != self.layers - 1:
                conv_layers.append(nn.BatchNorm2d(self.channels[i]))

            conv_layers.append(nn.GELU())

            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))

        return conv_layers

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.conv(x)
        x = self.output(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim = 32):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(output_dim = latent_dim,
                               use_batchnorm = True,
                               use_dropout=False)
        self.decoder = Decoder(input_dim = latent_dim,
                               use_batchnorm = True,
                               use_dropout = False)

    def forward(self, x):
        return self.decoder(self.encoder(x))
