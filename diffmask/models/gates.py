import numpy as np
import torch
import logging
from .distributions import RectifiedStreched, BinaryConcrete


class MLPGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.utils.weight_norm(torch.nn.Linear(input_size, hidden_size)),
            torch.nn.Tanh(),
            torch.nn.utils.weight_norm(torch.nn.Linear(hidden_size, 1, bias=bias)),
        )
        if bias:
            self.f[-1].bias.data[:] = 5.0

    def forward(self, *args):
        return self.f(torch.cat(args, -1))


class MLPMaxGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size, max_activation=10, bias=True):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.utils.weight_norm(torch.nn.Linear(input_size, hidden_size)),
            torch.nn.Tanh(),
            torch.nn.utils.weight_norm(torch.nn.Linear(hidden_size, 1, bias=bias)),
            torch.nn.Tanh(),
        )
        self.bias = torch.nn.Parameter(torch.tensor(5.0))
        self.max_activation = max_activation

    def forward(self, *args):
        return self.f(torch.cat(args, -1)) * self.max_activation + self.bias

class DiffMaskGateInput_FidDecoder(torch.nn.Module):
    def __init__(
        self,
        n_context,
        max_len,
        hidden_size: int,
        hidden_attention: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        gate_fn: torch.nn.Module = MLPMaxGate,
        gate_bias: bool = True,
        init_vector: torch.Tensor = None,
    ):
        super().__init__()
        self.n_context = n_context
        self.max_len = max_len

        self.g_hat = torch.nn.ModuleList(
            [
                gate_fn((hidden_size * max_len) * 2, hidden_attention, bias=gate_bias)
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, hidden_states, layer_pred):
        logging.info("gates.py: hidden_states[0] shape {}".format(hidden_states[0].shape))

        # hidden_states: bsz*n_context, len,  hidden_size
        temp, _ , hidden_size = hidden_states[0].shape
        bsz = int(temp / self.n_context)
        assert hidden_states[0].shape[1] == self.max_len
        reshaped_hidden_states = []
        for h in hidden_states:
            reshaped_hidden_states.append(h.view(bsz,self.n_context,-1))
        # hidden_states : bsz, n_context, len*hidden_size

        logits = torch.cat(
            [
                self.g_hat[i](reshaped_hidden_states[0], reshaped_hidden_states[i])
                for i in range(
                    (layer_pred + 1) if layer_pred is not None else len(reshaped_hidden_states)
                )
            ],
            -1,
        )
        logging.info("gates.py: logits.shape:{}".format(logits.shape))
        # logits: bsz, n_context, n_layer
        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(logits, 0.2), logits), l=-0.2, r=1.0,
        )


        gates_full = dist.rsample().cumprod(-1)
        expected_L0_full = dist.log_expected_L0().cumsum(-1)
        # gates: bsz, n_context
        gates = gates_full[..., -1]
        expected_L0 = expected_L0_full[..., -1]
        logging.info("gate.py: expected_L0 {}".format(expected_L0.shape))
        logging.info("gate.py: gates_full {}".format(gates_full.shape))
        logging.info("gate.py: gates {}".format(gates.shape))
    
        return (
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        )

class DiffMaskGateInput(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_attention: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        gate_fn: torch.nn.Module = MLPMaxGate,
        gate_bias: bool = True,
        placeholder: bool = False,
        init_vector: torch.Tensor = None,
    ):
        super().__init__()

        self.g_hat = torch.nn.ModuleList(
            [
                gate_fn(hidden_size * 2, hidden_attention, bias=gate_bias)
                for _ in range(num_hidden_layers)
            ]
        )

        if placeholder:
            self.placeholder = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(1, max_position_embeddings, hidden_size,)
                )
                if init_vector is None
                else init_vector.view(1, 1, hidden_size).repeat(
                    1, max_position_embeddings, 1
                )
            )
        else:
            self.register_buffer(
                "placeholder", torch.zeros((1, 1, hidden_size,)),
            )

    def forward(self, hidden_states, mask, layer_pred):
        logging.info("gates.py: hidden_states[0] shape {}".format(hidden_states[0].shape))
        # hidden_states: 1024
        # hidden_states : 1, 384(len), 1024
        # logits: 1, 384, 26(layer num)
        logits = torch.cat(
            [
                self.g_hat[i](hidden_states[0], hidden_states[i])
                for i in range(
                    (layer_pred + 1) if layer_pred is not None else len(hidden_states)
                )
            ],
            -1,
        )
        logging.info("gates.py: logits.shape:{}".format(logits.shape))
        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(logits, 0.2), logits), l=-0.2, r=1.0,
        )


        gates_full = dist.rsample().cumprod(-1)
        expected_L0_full = dist.log_expected_L0().cumsum(-1)

        gates = gates_full[..., -1]
        expected_L0 = expected_L0_full[..., -1]
        logging.info("gate.py: expected_L0 {}".format(expected_L0.shape))
        logging.info("gate.py: gates_full {}".format(gates_full.shape))
        logging.info("gate.py: gates {}".format(gates.shape))
        logging.info("gate.py: gates {}".format(gates))
        # print("gate.py: placeholder {}".format(self.placeholder[:, : hidden_states[0].shape[-2],].shape))
        # print("gate.py: placeholder {}".format(self.placeholder[:, : hidden_states[0].shape[-2],]))

        return (
            hidden_states[0] * gates.unsqueeze(-1)
            + self.placeholder[:, : hidden_states[0].shape[-2],]
            * (1 - gates).unsqueeze(-1),
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        )


class DiffMaskGateHidden(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_attention: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        gate_fn: torch.nn.Module = MLPMaxGate,
        gate_bias: bool = True,
        placeholder: bool = False,
        init_vector: torch.Tensor = None,
    ):
        super().__init__()

        self.g_hat = torch.nn.ModuleList(
            [
                gate_fn(hidden_size, hidden_attention, bias=gate_bias)
                for _ in range(num_hidden_layers)
            ]
        )

        if placeholder:
            self.placeholder = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.nn.init.xavier_normal_(
                            torch.empty(1, max_position_embeddings, hidden_size,)
                        )
                        if init_vector is None
                        else init_vector.view(1, 1, hidden_size).repeat(
                            1, max_position_embeddings, 1
                        )
                    )
                    for _ in range(num_hidden_layers)
                ]
            )
        else:
            self.register_buffer(
                "placeholder", torch.zeros((num_hidden_layers, 1, 1, hidden_size,)),
            )

    def forward(self, hidden_states, mask, layer_pred):

        if layer_pred is not None:
            logits = self.g_hat[layer_pred](hidden_states[layer_pred])
        else:
            logits = torch.cat(
                [self.g_hat[i](hidden_states[i]) for i in range(len(hidden_states))], -1
            )

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(logits, 0.2), logits), l=-0.2, r=1.0,
        )

        gates_full = dist.rsample()
        expected_L0_full = dist.log_expected_L0()

        gates = gates_full if layer_pred is not None else gates_full[..., :1]
        expected_L0 = (
            expected_L0_full if layer_pred is not None else expected_L0_full[..., :1]
        )

        return (
            hidden_states[layer_pred if layer_pred is not None else 0] * gates
            + self.placeholder[layer_pred if layer_pred is not None else 0][
                :,
                : hidden_states[layer_pred if layer_pred is not None else 0].shape[-2],
            ]
            * (1 - gates),
            gates.squeeze(-1),
            expected_L0.squeeze(-1),
            gates_full,
            expected_L0_full,
        )


class PerSampleGate(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        batch_size: int = 1,
        placeholder: bool = False,
        init_vector: torch.Tensor = None,
    ):
        super().__init__()

        self.logits = torch.nn.Parameter(
            torch.full((batch_size, max_position_embeddings, num_hidden_layers), 5.0)
        )

        if placeholder:
            self.placeholder = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(
                        batch_size,
                        num_hidden_layers,
                        max_position_embeddings,
                        hidden_size,
                    )
                )
                if init_vector is None
                else init_vector.view(1, num_hidden_layers, 1, hidden_size).repeat(
                    batch_size, 1, max_position_embeddings, 1
                )
            )
        else:
            self.register_buffer(
                "placeholder", torch.zeros((1, num_hidden_layers, 1, hidden_size))
            )


class PerSampleDiffMaskGate(PerSampleGate):
    def forward(self, hidden_states, mask, layer_pred):

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(self.logits, 0.2), self.logits),
            l=-0.2,
            r=1.0,
        )

        gates_full = dist.rsample()
        expected_L0_full = dist.log_expected_L0()

        gates = gates_full[..., layer_pred]
        expected_L0 = expected_L0_full[..., layer_pred]

        return (
            hidden_states[layer_pred] * gates.unsqueeze(-1)
            + self.placeholder[:, layer_pred, : hidden_states[layer_pred].shape[-2]]
            * (1 - gates).unsqueeze(-1),
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        )


class PerSampleREINFORCEGate(PerSampleGate):
    def forward(self, hidden_states, mask, layer_pred):

        dist = torch.distributions.Bernoulli(logits=self.logits)

        gates_full = dist.sample()
        expected_L0_full = dist.log_prob(1.0)

        gates = gates_full[..., layer_pred]
        expected_L0 = expected_L0_full[..., layer_pred]

        return (
            hidden_states[layer_pred] * gates.unsqueeze(-1)
            + self.placeholder[:, layer_pred, : hidden_states[layer_pred].shape[-2]]
            * (1 - gates).unsqueeze(-1),
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        )
