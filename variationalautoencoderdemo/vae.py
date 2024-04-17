"""Variational Autoencoder module."""

import keras

# from tensorflow.keras import layers, models


class VariationalAutoencoder(
    keras.Model
):  # pylint: disable=too-many-ancestors
    """Vae class."""

    def __init__(self, latent_dim):
        """Initialize class.

        Defines the latent dimension and initializes the encoder and decoder.
        It also sets up metrics in order to track the loss during training.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def build_encoder(self):
        """Define and build the encoder network.

        This methods maps the input to the latent space.
        """
        return 0, 1

    def build_decoder(self):
        """Define and build the decoder network.

        This methods tries to reconstructs the input from the latent space.
        """
        return 0

    def call(self, *args, **kwargs):
        """Pass input through the model."""
        z_mean, z_log_var = self.build_encoder()
        self.sampling(z_mean, z_log_var)
        reconstructed = self.build_decoder()
        return reconstructed

    def sampling(self, mean: float, variance: float) -> float:
        """Sample from the trained distribution."""
        dummy = mean * variance
        return dummy

    def add_metric(self, *args, **kwargs) -> None:
        """Add metrics."""

    def compute_output_shape(self, *args, **kwargs) -> None:
        """Compute the output shape."""

    def quantized_call(self, *args, **kwargs) -> None:
        """Init quantized call."""
