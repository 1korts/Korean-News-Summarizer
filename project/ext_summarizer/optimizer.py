import torch
from common.optimizer import Optimizer


def build_optim(params, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if params["use_gpu"]:
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            params["method"], params["lr"], params["max_grad_norm"],
            beta1=params["beta1"], beta2=params["beta2"],
            decay_method='noam',
            warmup_steps=params["warmup_steps"])

    optim.set_parameters(list(model.named_parameters()))

    return optim