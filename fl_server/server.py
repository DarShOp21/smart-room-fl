import flwr as fl
import torch

from fl_server.model.autoencoder import AutoEncoder
from fl_server.config import SERVER_ADDRESS, NUM_ROUNDS, MIN_CLIENTS


def get_initial_parameters():
    model = AutoEncoder()
    return [val.cpu().numpy() for val in model.state_dict().values()]


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(
        get_initial_parameters()
    ),
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    )
