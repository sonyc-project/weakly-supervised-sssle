from torch.optim import Adam, SGD


def get_optimizer(parameters, train_config):
    opt_config = train_config["training"]["optimizer"]
    opt_name = opt_config["name"]
    opt_params = opt_config["parameters"]

    if opt_name == "Adam":
        return Adam(parameters, **opt_params)
    elif opt_name == "SGD":
        return SGD(parameters, **opt_params)
    else:
        raise ValueError("Invalid optimizer: {}".format(opt_name))

