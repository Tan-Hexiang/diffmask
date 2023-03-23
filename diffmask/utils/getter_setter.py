import torch
from transformers import BertForSequenceClassification
from collections import defaultdict
import logging
from diffmask.models.question_answering_nq import (
load_nq, Dataset, Collator
)
from transformers import T5Tokenizer
from ..models.fid import FiDT5

def  fid_getter(model, passage_ids, passage_masks, target_ids, forward_fn=None):
    decoder_hidden_states = []
    encoder_embedding = []
    def get_hook(i):
        def hook(module, inputs, outputs=None):
            if 0 <= i < len(model.decoder.block):
                # bsz*n_context, len, hidden_size
                decoder_hidden_states.append(inputs[0])
            elif i == len(model.decoder.block):
                decoder_hidden_states.append(outputs[0])
        return hook
    
    def encoder_hook(module, inputs, outputs=None):
        encoder_embedding.append(outputs)

    handles = (
            [model.encoder.encoder.embed_tokens.register_forward_hook(encoder_hook)]+
            [
                block.register_forward_pre_hook(get_hook(i))
                for i, block in enumerate(model.decoder.block)
            ]
            + [
                model.decoder.block[-1].register_forward_hook(
                    get_hook(len(model.decoder.block))
                )
            ]
    )

    try:
        if forward_fn is None:
            logging.debug("passage_ids {}, passage_masks {}".format(passage_ids.shape, passage_masks.shape))
            # loss, logits, ...
            logits = model(input_ids = passage_ids, attention_mask = passage_masks, lm_labels = target_ids)[1]
        else:
            logits = forward_fn(passage_ids, passage_masks, 20)
    finally:
        for handle in handles:
            handle.remove()

    return logits, tuple(decoder_hidden_states), encoder_embedding[0]

def fid_setter(model, passage_ids, passage_masks, target_ids, new_hidden_states, forward_fn=None):
    # change encoder embedding output
    def hook(module, inputs, outputs=None):
            # because T5 use shared embed_tokens, we just mask encoder embedding layer
            # encoder embedding layer outputs bsz*n_context,passage_len,768
            # decoder embedding layer outputs bsz,target_len,768
            if outputs.shape == new_hidden_states.shape:
                return new_hidden_states

    handles = (
        [model.encoder.encoder.embed_tokens.register_forward_hook(hook)]
    )

    try:
        if forward_fn is None:
            logits = model(input_ids = passage_ids, attention_mask = passage_masks, labels = target_ids)[1]
        else:
            logits = forward_fn(passage_ids, passage_masks)
    finally:
        for handle in handles:
            handle.remove()

    return logits

def bert_getter(model, inputs_dict, forward_fn=None):

    hidden_states_ = []

    def get_hook(i):
        def hook(module, inputs, outputs=None):
            # print("hook {}: ".format(i))
            if i == 0:
                hidden_states_.append(outputs)
                # print(outputs.shape)
            elif 1 <= i <= len(model.bert.encoder.layer):
                hidden_states_.append(inputs[0])
                # print(inputs[0].shape)
            elif i == len(model.bert.encoder.layer) + 1:
                hidden_states_.append(outputs[0])
                # print(outputs[0].shape)

        return hook

    handles = (
        [model.bert.embeddings.word_embeddings.register_forward_hook(get_hook(0))]
        + [
            layer.register_forward_pre_hook(get_hook(i + 1))
            for i, layer in enumerate(model.bert.encoder.layer)
        ]
        + [
            model.bert.encoder.layer[-1].register_forward_hook(
                get_hook(len(model.bert.encoder.layer) + 1)
            )
        ]
    )

    try:
        if forward_fn is None:
            outputs = model(**inputs_dict)
        else:
            outputs = forward_fn(**inputs_dict)
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)


def bert_setter(model, inputs_dict, hidden_states, forward_fn=None):

    hidden_states_ = []

    def get_hook(i):
        def hook(module, inputs, outputs=None):
            if i == 0:
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return hidden_states[i]
                else:
                    hidden_states_.append(outputs)

            elif 1 <= i <= len(model.bert.encoder.layer):
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return (hidden_states[i],) + inputs[1:]
                else:
                    hidden_states_.append(inputs[0])

            elif i == len(model.bert.encoder.layer) + 1:
                if hidden_states[i] is not None:
                    hidden_states_.append(hidden_states[i])
                    return (hidden_states[i],) + outputs[1:]
                else:
                    hidden_states_.append(outputs[0])

        return hook

    handles = (
        [model.bert.embeddings.word_embeddings.register_forward_hook(get_hook(0))]
        + [
            layer.register_forward_pre_hook(get_hook(i + 1))
            for i, layer in enumerate(model.bert.encoder.layer)
        ]
        + [
            model.bert.encoder.layer[-1].register_forward_hook(
                get_hook(len(model.bert.encoder.layer) + 1)
            )
        ]
    )

    try:
        if forward_fn is None:
            outputs = model(**inputs_dict)
        else:
            outputs = forward_fn(**inputs_dict)
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)




def gru_getter(model, inputs_dict):

    hidden_states_ = []

    def get_hook():
        def hook(module, inputs, outputs):
            hidden_states_.append(outputs)

        return hook

    handles = [model.emb.register_forward_hook(get_hook())]

    try:
        outputs = model(**inputs_dict)
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)


def gru_setter(model, inputs_dict, hidden_states):

    hidden_states_ = []

    def get_hook():
        def hook(module, inputs, outputs):
            if hidden_states[0] is not None:
                hidden_states_.append(hidden_states[0])
                return hidden_states[0]
            else:
                hidden_states_.append(outputs)

        return hook

    handles = [model.emb.register_forward_hook(get_hook())]

    try:
        outputs = model(**inputs_dict)
    finally:
        for handle in handles:
            handle.remove()

    return outputs, tuple(hidden_states_)


def toy_getter(model, inputs_dict):

    hidden_states_ = []
    handles = [
        model.embedding_query.register_forward_hook(
            lambda module, inputs, outputs: hidden_states_.append(outputs)
        ),
        model.embedding_input.register_forward_hook(
            lambda module, inputs, outputs: hidden_states_.append(outputs)
        ),
        model.encoder.register_forward_hook(
            lambda module, inputs, outputs: hidden_states_.append(outputs)
        ),
    ]

    outputs = model(**inputs_dict)
    for handle in handles:
        handle.remove()

    return outputs, tuple(hidden_states_)


def toy_setter(model, inputs_dict, hidden_states):

    handles = [
        model.embedding_query.register_forward_hook(
            lambda module, inputs, outputs: hidden_states[0]
            if hidden_states[0] is not None
            else None
        ),
        model.embedding_input.register_forward_hook(
            lambda module, inputs, outputs: hidden_states[1]
            if hidden_states[1] is not None
            else None
        ),
        model.encoder.register_forward_hook(
            lambda module, inputs, outputs: hidden_states[2]
            if hidden_states[2] is not None
            else None
        ),
    ]

    outputs = model(**inputs_dict)
    for handle in handles:
        handle.remove()

    return outputs, None

def test_fid_getter():
   
    tokenizer = T5Tokenizer.from_pretrained('t5-base', return_dict=False)
    collator = Collator(200, tokenizer)
    data = load_nq("/data/tanhexiang/tevatron/tevatron/data_nq/result100/fid.nq.small.jsonl")
    dataset = Dataset(data, n_context=100, passages_source_path="/data/tanhexiang/CF_QA/data/wikipedia_split/psgs_w100.tsv")
    dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=2, shuffle=True, collate_fn=collator
            )
    batch = next(iter(dataloader))
    (index, target_ids, target_mask, passage_ids, passage_mask) = batch
    
    model = FiDT5.from_pretrained("/data/tanhexiang/CF_QA/models/reader/nq_reader_base")
    model.eval()
    outputs = model(passage_ids, passage_mask, lm_labels = target_ids, output_hidden_states=True)
    print(type(outputs))
    print(len(outputs))
    decoder_hidden_states = outputs[3]
    for i, s in enumerate(decoder_hidden_states):
        print("i {} s {}".format(i,s))
        exit()

def test_bert_getter():

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").cuda()

    inputs_dict = {
        "input_ids": torch.tensor(
            [[101, 1037, 4010, 1010, 6057, 1010, 11973, 2143, 1012, 102, 0, 0, 0,]]
        ).cuda(),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,]]
        ).cuda(),
        "labels": torch.tensor([1]).cuda(),
    }

    model.bert.encoder.output_hidden_states = True

    outputs_orig = model(**inputs_dict)

    outputs_orig = (
        outputs_orig[0],
        outputs_orig[1],
        (model.bert.embeddings.word_embeddings(inputs_dict["input_ids"]),)
        + outputs_orig[2],
    )

    model.bert.encoder.output_hidden_states = False

    outputs = bert_getter(model, inputs_dict)

    assert (outputs_orig[0] == outputs[0][0]).all()

    assert (outputs_orig[1] == outputs[0][1]).all()

    assert all((a == b).all() for a, b in zip(outputs_orig[2], outputs[1]))


def test_bert_setter():

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").cuda()

    inputs_dict = {
        "input_ids": torch.tensor(
            [[101, 1037, 4010, 1010, 6057, 1010, 11973, 2143, 1012, 102, 0, 0, 0,]]
        ).cuda(),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,]]
        ).cuda(),
        "labels": torch.tensor([1]).cuda(),
    }

    outputs, hidden_states = bert_getter(model, inputs_dict)

    for i, h in enumerate(hidden_states):
        outputs_, hidden_states_ = bert_setter(
            model,
            inputs_dict,
            [None] * i + [h * 0] + [None] * (len(hidden_states) - 1 - i),
        )
        assert all((a != b).all() for a, b in zip(outputs, outputs_))

if __name__=="__main__":
    test_fid_getter()