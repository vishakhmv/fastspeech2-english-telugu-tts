import numpy as np
import torch


class ScheduledOptim:
    def __init__(self, model, train_config, model_config, current_step=0):
        optimizer_config = train_config["optimizer"]
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=tuple(optimizer_config["betas"]),
            eps=optimizer_config["eps"],
            weight_decay=optimizer_config["weight_decay"],
        )
        self.n_warmup_steps = optimizer_config["warm_up_step"]
        self.anneal_steps = optimizer_config["anneal_steps"]
        self.anneal_rate = optimizer_config["anneal_rate"]
        self.current_step = current_step
        self.init_lr = float(model_config["transformer"]["encoder_hidden"]) ** -0.5

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def current_lr(self):
        return self._optimizer.param_groups[0]["lr"]

    def update_learning_rate(self):
        self.current_step += 1
        lr = float(self.init_lr * self._get_lr_scale())
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def zero_grad(self):
        self._optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state):
        self._optimizer.load_state_dict(state)
