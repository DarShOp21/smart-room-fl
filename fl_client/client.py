import flwr as fl
import torch
import torch.nn as nn
import numpy as np

from fl_client.model.autoencoder import AutoEncoder
from fl_client.synthetic.generator import generate_dataset
from fl_client.trust.trust_score import (
    compute_reconstruction_error,
    compute_trust_score
)
from shared.config import SERVER_ADDRESS

# Device configuration
DEVICE = torch.device("cpu")


class SmartRoomClient(fl.client.NumPyClient):

    def __init__(self):
        # Initialize model
        self.model = AutoEncoder().to(DEVICE)

        # Loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Generate synthetic smart-room behavior dataset
        dataset = generate_dataset(num_samples=50, suspicious_ratio=0.2)

        self.x_train = torch.tensor(
            dataset,
            dtype=torch.float32
        ).to(DEVICE)

    # ----------------------------------------------------
    # Flower: send model parameters to the server
    # ----------------------------------------------------
    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.state_dict().values()]

    # ----------------------------------------------------
    # Flower: receive parameters from server
    # ----------------------------------------------------
    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in params_dict
        }

        self.model.load_state_dict(state_dict, strict=True)

    # ----------------------------------------------------
    # Local training step
    # ----------------------------------------------------
    def fit(self, parameters, config):

        # Update model with global parameters
        self.set_parameters(parameters)

        self.model.train()

        for _ in range(5):  # local epochs
            self.optimizer.zero_grad()

            output = self.model(self.x_train)

            loss = self.loss_fn(output, self.x_train)

            loss.backward()

            self.optimizer.step()

        # Return updated weights
        return self.get_parameters(config), len(self.x_train), {}

    # ----------------------------------------------------
    # Evaluation step (trust score calculation)
    # ----------------------------------------------------
    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        self.model.eval()

        with torch.no_grad():

            reconstructed = self.model(self.x_train)

            error = compute_reconstruction_error(
                self.x_train,
                reconstructed
            )

            trust = compute_trust_score(error)

            print("\n----- Client Evaluation -----")
            print("Reconstruction Error:", error)
            print("Trust Score:", trust)
            print("-----------------------------\n")

        return float(error), len(self.x_train), {"trust_score": trust}


# --------------------------------------------------------
# Start the Flower client
# --------------------------------------------------------
if __name__ == "__main__":

    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=SmartRoomClient().to_client(),
    )