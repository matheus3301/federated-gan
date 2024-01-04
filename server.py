import flwr as fl

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 256,
        "current_round": server_round,
        "local_epochs": 10,
    }
    return config

strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=fit_config,
)

fl.server.start_server(
  server_address="0.0.0.0:8080",
  config=fl.server.ServerConfig(num_rounds=2),
  strategy=strategy,
)