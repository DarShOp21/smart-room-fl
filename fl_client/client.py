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

DEVICE = torch.device("cpu")

class SmartRoomClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = AutoEncoder().to(DEVICE)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)

        # Generate only normal data for training
        dataset = generate_dataset(num_samples=2000, suspicious_ratio=0.0)
        self.x_train = torch.tensor(dataset, dtype=torch.float32).to(DEVICE)

        # Validation set from the same normal distribution
        val_dataset = generate_dataset(num_samples=100, suspicious_ratio=0.0)
        self.x_val = torch.tensor(val_dataset, dtype=torch.float32).to(DEVICE)

        # Will be set after first fit
        self.error_threshold = 0.15  # fallback

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        # Local training (more epochs than before)
        epochs = 10
        for _ in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(self.x_train)
            loss = self.loss_fn(output, self.x_train)
            loss.backward()
            self.optimizer.step()

        # After training, compute reconstruction errors on training set to set threshold
        self.model.eval()
        with torch.no_grad():
            recon = self.model(self.x_train)
            errors = torch.mean((self.x_train - recon) ** 2, dim=1).cpu().numpy()
            self.error_threshold = np.percentile(errors, 90)  # 90th percentile

        return self.get_parameters(config), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        with torch.no_grad():
            recon = self.model(self.x_val)
            error = compute_reconstruction_error(self.x_val, recon)
            trust = compute_trust_score(error, normal_error_threshold=self.error_threshold)

            print("\n----- Client Evaluation -----")
            print(f"Reconstruction Error: {error:.4f}")
            print(f"Error Threshold: {self.error_threshold:.4f}")
            print(f"Trust Score: {trust:.4f}")
            print("-----------------------------\n")

        return float(error), len(self.x_val), {"trust_score": trust}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=SmartRoomClient().to_client(),
    )