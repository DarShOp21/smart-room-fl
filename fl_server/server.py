import flwr as fl
import torch

from fl_server.model.autoencoder import AutoEncoder
from fl_server.config import SERVER_ADDRESS, NUM_ROUNDS
from fl_server.strategy.fedavg import get_fedavg_strategy


def get_initial_parameters():
    """
    Initialize the global model parameters that will be
    sent to clients at the start of training.
    """
    model = AutoEncoder()
    return [val.cpu().numpy() for val in model.state_dict().values()]


if __name__ == "__main__":

    # Convert model weights into Flower parameters
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_initial_parameters()
    )

    # Load the strategy
    strategy = get_fedavg_strategy(initial_parameters)

    # Start the federated server
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    )