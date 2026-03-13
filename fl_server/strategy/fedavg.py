import flwr as fl
import numpy as np

from flwr.server.strategy import FedAvg
from fl_server.cloud_client import upload_model, upload_trust


class TrustWeightedFedAvg(FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        if not results:
            return None, {}

        weights = []
        trusts = []

        for client, fit_res in results:

            params = fl.common.parameters_to_ndarrays(
                fit_res.parameters
            )

            trust = fit_res.metrics.get("trust_score", 0.5)

            weights.append(params)
            trusts.append(trust)

        trusts = np.array(trusts)

        # normalize trust weights
        if trusts.sum() == 0:
            trusts = np.ones_like(trusts)

        trusts = trusts / trusts.sum()

        # weighted aggregation
        aggregated = []

        for layer_i in range(len(weights[0])):

            layer_stack = np.array(
                [client[layer_i] for client in weights]
            )

            weighted_layer = np.tensordot(
                trusts,
                layer_stack,
                axes=1
            )

            aggregated.append(weighted_layer)

        aggregated_parameters = fl.common.ndarrays_to_parameters(
            aggregated
        )

        # Upload global model to cloud
        upload_model(aggregated)

        return aggregated_parameters, {}


def aggregate_metrics(metrics):

    trust_scores = []

    for _, m in metrics:

        if "trust_score" in m:
            trust_scores.append(m["trust_score"])

    if len(trust_scores) > 0:
        avg_trust = sum(trust_scores) / len(trust_scores)
    else:
        avg_trust = 0

    print("Average Trust:", avg_trust)

    upload_trust(avg_trust)

    return {"avg_trust": avg_trust}


def get_fedavg_strategy(initial_parameters):

    strategy = TrustWeightedFedAvg(

        fraction_fit=1.0,

        min_fit_clients=1,

        min_available_clients=1,

        min_evaluate_clients=1,

        evaluate_metrics_aggregation_fn=aggregate_metrics,

        initial_parameters=initial_parameters,
    )

    return strategy