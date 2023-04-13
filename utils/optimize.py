from torch.optim import Adam


def configure_optimizers(net, args):
    """
    Separate parameters for the main optimizer and auxiliary optimizer.
    Return two optimizers.
    """
    parameters = {
        n for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in net.named_parameters()
        if n.endswith("quantiles") and p.requires_grad
    }

    # Make sure we dont have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0     # All the requires_grad params ?? DUCANH

    optimizer = Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate
    )

    aux_optimizer = Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate
    )

    return optimizer, aux_optimizer