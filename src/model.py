from __future__ import division

import torch
import torch.nn as nn


class VariationalAutoEncoder(nn.Module):
    """Base class for variational autoencoder models.

    This class defines the VAE architecture and implements the reparameterization
    trick for sampling from the latent space.
    """

    def __init__(self) -> None:
        super().__init__()
        self.apply(self.weights_init)

        # These will be implemented by subclasses
        self.encoder = None
        self.mu = None
        self.logvar = None
        self.linear2 = None
        self.decoder = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Perform the reparameterization trick to enable backpropagation through random sampling.

        Args:
            mu: Mean of the latent Gaussian.
            logvar: Log variance of the latent Gaussian.

        Returns:
            A sample from the latent Gaussian.
        """
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x: Input tensor.

        Returns:
            Tuple containing:
                z: Latent vector.
                x_hat: Reconstructed input.
                mu: Mean vector of the latent Gaussian.
                std: Standard deviation of the latent Gaussian.
        """
        encoder_out = self.encoder(x)

        mu = self.mu(encoder_out.flatten(1, -1))
        logvar = self.logvar(encoder_out.flatten(1, -1))
        std = torch.exp(logvar / 2)
        z = self.reparameterize(mu, logvar)

        out = self.linear2(z)

        x_hat = self.decoder(out.view(encoder_out.size()))

        return z, x_hat, mu, std

    @staticmethod
    def weights_init(m: nn.Module) -> None:
        """Initialize network weights using standard initialization techniques.

        Args:
            m: A PyTorch module whose weights will be initialized.
        """
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


# class VariationalAutoEncoderRawData(VariationalAutoEncoder):
#     """Variational Autoencoder implementation for raw image data.

#     This VAE architecture is designed to work with grayscale images through a
#     series of convolutional layers.
#     """

#     def __init__(self, inplanes: int = 32, latent_dim: int = 16) -> None:
#         """Initialize the VAE for raw image data.

#         Args:
#             inplanes: Base number of convolutional filters.
#             latent_dim: Dimension of the latent space.
#         """
#         super().__init__()

#         # Create encoder and decoder components
#         self._create_network(inplanes, latent_dim)

#     def _create_conv_block(
#         self,
#         in_dim: int,
#         out_dim: int,
#         filter_size: int = 3,
#         stride: int = 1,
#         padding: int = 1,
#     ) -> nn.Sequential:
#         """Create a standard convolutional block.

#         Args:
#             in_dim: Number of input channels.
#             out_dim: Number of output channels.
#             filter_size: Size of convolutional kernel.
#             stride: Stride of convolution.
#             padding: Padding size.

#         Returns:
#             A sequential container of convolutional block layers.
#         """
#         return nn.Sequential(
#             nn.Conv2d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
#             nn.InstanceNorm2d(out_dim),
#             nn.LeakyReLU(0.02),
#         )

#     def _create_transpose_conv_block(
#         self,
#         in_dim: int,
#         out_dim: int,
#         filter_size: int = 3,
#         stride: int = 1,
#         padding: int = 1,
#     ) -> nn.Sequential:
#         """Create a standard transpose convolutional block.

#         Args:
#             in_dim: Number of input channels.
#             out_dim: Number of output channels.
#             filter_size: Size of convolutional kernel.
#             stride: Stride of convolution.
#             padding: Padding size.

#         Returns:
#             A sequential container of transpose convolutional block layers.
#         """
#         return nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_dim, out_dim, filter_size, stride=stride, padding=padding
#             ),
#             nn.InstanceNorm2d(out_dim),
#             nn.LeakyReLU(0.02),
#         )

#     def _create_network(self, inplanes: int, latent_dim: int) -> None:
#         """Create the encoder and decoder networks.

#         Args:
#             inplanes: Base number of convolutional filters.
#             latent_dim: Dimension of the latent space.
#         """
#         # Calculate the size of flattened features after encoder
#         flat_features = inplanes * 4 * 4 * 4

#         # Create encoder
#         self.encoder = self._create_encoder(inplanes)

#         # Create latent space projections
#         self.mu = nn.Linear(flat_features, latent_dim)
#         self.logvar = nn.Linear(flat_features, latent_dim)
#         self.linear2 = nn.Linear(latent_dim, flat_features)

#         # Create decoder
#         self.decoder = self._create_decoder(inplanes)

#     def _create_encoder(self, inplanes: int) -> nn.Sequential:
#         """Create the encoder network."""
#         return nn.Sequential(
#             self._create_conv_block(1, inplanes),
#             self._create_conv_block(inplanes, inplanes),
#             nn.MaxPool2d(2, 2),  # size = [batch,inplanes,image_size/2,image_size/2]
#             self._create_conv_block(inplanes, inplanes * 2),
#             self._create_conv_block(inplanes * 2, inplanes * 2),
#             nn.MaxPool2d(2, 2),  # size = [batch,inplanes*2,image_size/4,image_size/4]
#             self._create_conv_block(inplanes * 2, inplanes * 4),
#             self._create_conv_block(inplanes * 4, inplanes * 4),
#             nn.MaxPool2d(2, 2),  # size = [batch,inplanes*4,image_size/8,image_size/8]
#             self._create_conv_block(inplanes * 4, inplanes * 4),
#             self._create_conv_block(inplanes * 4, inplanes * 4),
#             nn.MaxPool2d(2, 2),  # size = [batch,inplanes*4,image_size/16,image_size/16]
#             self._create_conv_block(inplanes * 4, inplanes * 4),
#             self._create_conv_block(inplanes * 4, inplanes * 4),
#             nn.MaxPool2d(2, 2),  # size = [batch,inplanes*4,image_size/32,image_size/32]
#         )

#     def _create_decoder(self, inplanes: int) -> nn.Sequential:
#         """Create the decoder network."""
#         return nn.Sequential(
#             nn.UpsamplingNearest2d(scale_factor=2),
#             self._create_transpose_conv_block(inplanes * 4, inplanes * 4),
#             self._create_transpose_conv_block(inplanes * 4, inplanes * 4),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             self._create_transpose_conv_block(inplanes * 4, inplanes * 4),
#             self._create_transpose_conv_block(inplanes * 4, inplanes * 4),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             self._create_transpose_conv_block(inplanes * 4, inplanes * 4),
#             self._create_transpose_conv_block(inplanes * 4, inplanes * 2),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             self._create_transpose_conv_block(inplanes * 2, inplanes * 2),
#             self._create_transpose_conv_block(inplanes * 2, inplanes),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             self._create_transpose_conv_block(inplanes, inplanes),
#             nn.Conv2d(inplanes, 1, 3, 1, 1),
#             # nn.Sigmoid()  # Uncomment if needed for your specific use case
#         )


class VariationalAutoEncoderRawData(VariationalAutoEncoder):
    """Variational Autoencoder implementation for raw image data.

    This VAE architecture is designed to work with grayscale images through a
    series of convolutional layers.
    """

    def __init__(self, inplanes: int = 32, latent_dim: int = 16):
        super().__init__()

        def building_blocks(in_dim, out_dim, filter_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
                nn.InstanceNorm2d(out_dim),
                nn.LeakyReLU(0.02),
            )

        def building_blocks_trans(in_dim, out_dim, filter_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_dim, out_dim, filter_size, stride=stride, padding=padding
                ),
                nn.InstanceNorm2d(out_dim),
                nn.LeakyReLU(0.02),
            )

        self.encoder = nn.Sequential(
            building_blocks(1, inplanes, 3, 1, 1),
            building_blocks(inplanes, inplanes, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # size = [batch,16,image_size/2,image_size/2]
            building_blocks(inplanes, inplanes * 2, 3, 1, 1),
            building_blocks(inplanes * 2, inplanes * 2, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # size = [batch,32,image_size/4,image_size/4]
            building_blocks(inplanes * 2, inplanes * 4, 3, 1, 1),
            building_blocks(inplanes * 4, inplanes * 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # size = [batch,64,image_size/8,image_size/8]
            building_blocks(inplanes * 4, inplanes * 4, 3, 1, 1),
            building_blocks(inplanes * 4, inplanes * 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # size = [batch,64,image_size/16,image_size/16]
            building_blocks(inplanes * 4, inplanes * 4, 3, 1, 1),
            building_blocks(inplanes * 4, inplanes * 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # size = [batch,64,image_size/32,image_size/32]
        )

        self.mu = nn.Sequential(nn.Linear(inplanes * 4 * 4 * 4, latent_dim))

        self.logvar = nn.Sequential(nn.Linear(inplanes * 4 * 4 * 4, latent_dim))

        self.linear2 = nn.Sequential(nn.Linear(latent_dim, inplanes * 4 * 4 * 4))

        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            building_blocks_trans(inplanes * 4, inplanes * 4, 3, 1, 1),
            building_blocks_trans(inplanes * 4, inplanes * 4, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            building_blocks_trans(inplanes * 4, inplanes * 4, 3, 1, 1),
            building_blocks_trans(inplanes * 4, inplanes * 4, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            building_blocks_trans(inplanes * 4, inplanes * 4, 3, 1, 1),
            building_blocks_trans(inplanes * 4, inplanes * 2, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            building_blocks_trans(inplanes * 2, inplanes * 2, 3, 1, 1),
            building_blocks_trans(inplanes * 2, inplanes, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            building_blocks_trans(inplanes, inplanes, 3, 1, 1),
            nn.Conv2d(32, 1, 3, 1, 1),
            # nn.Sigmoid()
        )
