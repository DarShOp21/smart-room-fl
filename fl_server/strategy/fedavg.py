import flwr as fl
from flwr.server.strategy import FedAvg


def get_fedavg_strategy(initial_parameters):
    """
    Configure the FedAvg strategy for federated learning.
    """

    strategy = FedAvg(

        # fraction of clients used for training
        fraction_fit=1.0,

        # fraction used for evaluation
        fraction_evaluate=1.0,

        # minimum clients required for training
        min_fit_clients=1,

        # minimum clients required for evaluation
        min_evaluate_clients=1,

        # minimum clients that must be connected
        min_available_clients=1,

        # initial model parameters
        initial_parameters=initial_parameters,
    )

    return strategy