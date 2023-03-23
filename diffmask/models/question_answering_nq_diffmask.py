import torch
import numpy as np
import pytorch_lightning as pl
import logging
from transformers import (
    get_constant_schedule_with_warmup,
    get_constant_schedule,
)
from .question_answering_nq import (
    QuestionAnsweringNQ,
    FidQuestionAnsweringNQ,
)
from .gates import (
    DiffMaskGateInput,
    DiffMaskGateHidden,
    PerSampleDiffMaskGate,
    PerSampleREINFORCEGate,
    MLPMaxGate,
    MLPGate,
    DiffMaskGateInput_FidDecoder,
    MLPMaxGate_FidDecoder
)
from ..optim.lookahead import LookaheadRMSprop
from ..utils.getter_setter import (
    fid_getter,
    fid_setter,
)
from ..utils.util import accuracy_precision_recall_f1


class QuestionAnsweringNQDiffMask(QuestionAnsweringNQ):
    def __init__(self, hparams):
        super().__init__(hparams)

        for p in self.parameters():
            p.requires_grad_(False)

    def training_step(self, batch, batch_idx=None, optimizer_idx=None, name="train"):

        (index, target_ids, target_mask, passage_ids, passage_masks) = batch
        logging.debug("index{}, target_ids{}, target_mask{}, passage_ids{}, passage_masks{}".format(index.shape, target_ids.shape, target_mask.shape, passage_ids.shape, passage_masks.shape))

        (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_pred,
        ) = self.forward_explainer(batch)

        logging.debug("logits:{}".format(logits.shape))
        logging.debug("layer_pred:{}".format(layer_pred))
        logging.debug("layer_pred:{}".format(layer_pred))
        
        loss_c = (
            torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=logits_orig),
                torch.distributions.Categorical(logits=logits),
            )
            - self.hparams.eps
        )

        loss_g = expected_L0.sum(-1) 

        loss = self.alpha[layer_pred] * loss_c.mean(-1) + loss_g.mean(-1)



        l0 = expected_L0.exp().sum(-1)

        outputs_dict = {
            "loss_c": loss_c.mean(-1).mean(-1),
            "loss_g": loss_g.mean(-1),
            "alpha": self.alpha[layer_pred].mean(-1),
            "l0": l0.mean(-1),
            "layer_pred": layer_pred,
        }

        outputs_dict = {
            "loss": loss.mean(-1),
            **outputs_dict,
            "log": outputs_dict,
            "progress_bar": outputs_dict,
        }

        outputs_dict = {
            "{}{}".format("" if self.training else "val_", k): v
            for k, v in outputs_dict.items()
        }

        tb = self.logger.experiment
        tb.add_scalar("loss/"+name, loss.mean(-1))
        tb.add_scalar("l0/"+name, l0.mean(-1))
        tb.add_scalar("loss_c/"+name, loss_c.mean(-1))
        tb.add_scalar("loss_g/"+name, loss_g.mean(-1))
        tb.add_scalar("alpha/"+name, self.alpha[layer_pred].mean(-1))

        return outputs_dict
    
    def validation_step(self, batch, batch_idx=None):
        return self.training_step(batch, batch_idx,name="val")

    def validation_epoch_end(self, outputs):
        logging.debug(outputs)
        loss = sum([e['val_loss'] for e in outputs ]) / len(outputs)
        loss_c = sum([e['val_loss_c'] for e in outputs])/ len(outputs)
        loss_g = sum([e['val_loss_g'] for e in outputs])/ len(outputs)
        l0 = sum([e['val_l0'] for e in outputs])/len(outputs)
        alpha = sum([e['val_alpha'] for e in outputs])/len(outputs)
        tb = self.logger.experiment
        tb.add_scalar("val_loss/val_epoch", loss)
        tb.add_scalar("val_l0/val_epoch", l0)
        tb.add_scalar("val_loss_c/val_epoch", loss_c)
        tb.add_scalar("val_loss_g/val_epoch", loss_g)
        tb.add_scalar("val_alpha/val_epoch", alpha)
        return {
            "val_loss":loss,
            "val_loss_c": loss_c,
            "val_loss_g": loss_g,
            "val_alpha":alpha,
            "val_l0": l0,
            }


    def configure_optimizers(self):
        optimizers = [
            LookaheadRMSprop(
                params=[
                    {
                        "params": self.gate.g_hat.parameters(),
                        "lr": self.hparams.learning_rate,
                    },
                    {
                        "params": self.placeholder.parameters()
                        if isinstance(self.placeholder, torch.nn.ParameterList)
                        else [self.placeholder],
                        "lr": self.hparams.learning_rate_placeholder,
                    },
                ],
                centered=True,
            ),
            LookaheadRMSprop(
                params=[self.alpha]
                if isinstance(self.alpha, torch.Tensor)
                else self.alpha.parameters(),
                lr=self.hparams.learning_rate_alpha,
            ),
        ]

        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 24 * 50),
                "interval": "step",
            },
            get_constant_schedule(optimizers[1]),
        ]
        return optimizers, schedulers

    def optimizer_step(
        self,
        current_epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
    ):
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

        elif optimizer_idx == 1:
            for i in range(len(self.alpha)):
                if self.alpha[i].grad:
                    self.alpha[i].grad *= -1

            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

            for i in range(len(self.alpha)):
                self.alpha[i].data = torch.where(
                    self.alpha[i].data < 0,
                    torch.full_like(self.alpha[i].data, 0),
                    self.alpha[i].data,
                )
                self.alpha[i].data = torch.where(
                    self.alpha[i].data > 200,
                    torch.full_like(self.alpha[i].data, 200),
                    self.alpha[i].data,
                )


class FidQuestionAnsweringNQDiffMask(
    QuestionAnsweringNQDiffMask, FidQuestionAnsweringNQ,
):
    def __init__(self, hparams):
        super().__init__(hparams)
        # 
        self.target_len = 20
        # lagrange multiplier
        self.alpha = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.ones(()))
                for _ in range(self.net.config.num_layers + 1)
            ]
        )
        # mask vector
        if hparams.placeholder:
            self.placeholder = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(1, 1, self.net.config.d_model*self.hparams.passage_len)
                )
            )
        else:
            self.register_buffer(
                "placeholder", torch.zeros((1, 1, self.net.config.d_model*self.max_len,)),
            )

        gate = DiffMaskGateInput_FidDecoder

        self.gate = gate(
            n_context=hparams.n_context,
            target_len = self.hparams.target_len,
            passage_len=self.hparams.passage_len,
            hidden_size=self.net.config.d_model,
            hidden_attention=self.net.config.d_model // 4,
            # decoder 12layer input + last layer output
            num_hidden_layers=self.net.config.num_layers + 1,
            max_position_embeddings=1,
            gate_fn=MLPMaxGate_FidDecoder,
            gate_bias=hparams.gate_bias,
            init_vector=None,
        )

        for name, p in self.named_parameters():
            if p.requires_grad:
                # print("requires_grad: {}".format(name))
                logging.debug("requires_grad: {} {}".format(name, p.shape))
            else:
                logging.debug("close grad of {}".format(name))

    def forward_explainer(
        self,
        batch,
        layer_pred=None,
        attribution=False,
    ):
        # passage_ids: bsz, n_context, len
        # 训练解释的时候不更新net
        self.net.eval()
        (index, target_ids, target_mask, passage_ids, passage_mask) = batch

        logits_orig, decoder_hidden_states, encoder_embedding = fid_getter(
            self.net, passage_ids, passage_mask, target_ids
        )
        logging.debug("logits_orig:{}".format(logits_orig))
        logging.debug("decoder_hidden_states: {} {}".format(type(decoder_hidden_states),len(decoder_hidden_states)))
        logging.debug("encoder_embeddings: {} {}".format(type(encoder_embedding),encoder_embedding.shape))

        if layer_pred is None:
            if self.hparams.layer_pred == -1:
                layer_pred = torch.randint(len(decoder_hidden_states), ()).item()
            else:
                layer_pred = self.hparams.layer_pred
        (
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        ) = self.gate(
            hidden_states=decoder_hidden_states,
            layer_pred=None if attribution else layer_pred,
        )

        if attribution:
            return logits_orig, expected_L0_full
        else:
            # mask encoder embedding之后的向量 
            # bsz*n_context,len,dim
            temp,passage_len,dim = encoder_embedding.shape
            bsz = int(temp/self.hparams.n_context)
            new_encoder_embedding = encoder_embedding.view(bsz, self.hparams.n_context, -1)
            # bsz,n_context,len*dim
            new_encoder_embedding = new_encoder_embedding * gates.unsqueeze(-1) + self.placeholder *(1 - gates).unsqueeze(-1)
            new_encoder_embedding = new_encoder_embedding.view(bsz*self.hparams.n_context, passage_len,dim)
            logging.debug("new_encoder_embedding shape :{}".format(new_encoder_embedding.shape))
            logging.debug("new_encoder_embedding detail: {}".format(new_encoder_embedding))
         
            logits = fid_setter(
                self.net, passage_ids, passage_mask, target_ids, new_hidden_states=new_encoder_embedding,
            )
        return (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_pred,
        )
