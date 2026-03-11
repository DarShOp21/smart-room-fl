import flwr as fl
from flwr.server.strategy import FedAvg


def get_fedavg_strategy(initial_parameters):
    """
    Configure the FedAvg strategy for federated learning.
    """

    strategy = FedAvg(

        # fraction of clients used for training
        fraction_fit=1.0,

        # # fraction used for evaluation
        # fraction_evaluate=1.0,

        # minimum clients required for training
        min_fit_clients=1,

        # # minimum clients required for evaluation
        # min_evaluate_clients=1,

        # minimum clients that must be connected
        min_available_clients=1,

        # function to aggregate evaluation metrics
        evaluate_metrics_aggregation_fn=aggregate_metrics,

        # initial model parameters
        initial_parameters=initial_parameters,
    )

    return strategy

def aggregate_metrics(metrics):

    trust_scores = []

    for _, m in metrics:
        if "trust_score" in m:
            trust_scores.append(m["trust_score"])

    if len(trust_scores) > 0:
        avg_trust = sum(trust_scores) / len(trust_scores)
    else:
        avg_trust = None

    print("Average client trust:", avg_trust)

    return {"avg_trust": avg_trust}