import json
from collections import OrderedDict
import random
from tqdm.auto import tqdm
import torch
import logging
import pandas as pd
import pytorch_lightning as pl
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    T5Tokenizer,
    get_constant_schedule_with_warmup,
    get_constant_schedule,
)
from ..utils.util import accuracy_precision_recall_f1
from ..utils.jsonl import load_all_jsonl
from .fid import FiDT5


class QuestionAnsweringNQ(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # fid固定使用t5的tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base', return_dict=False)
        self.collator = Collator(self.hparams.text_maxlength, self.tokenizer)
    def prepare_data(self):
        # assign to use in dataloaders
        if (
            not hasattr(self, "train_dataset")
        ) and self.training:
            train_data = load_nq(self.hparams.train_filename)
            self.train_dataset =  Dataset(train_data, n_context= self.hparams.n_context, passages_source_path=self.hparams.passages_source_path)
        if not hasattr(self, "val_dataset") :
            val_data = load_nq(self.hparams.val_filename)
            self.val_dataset = Dataset(val_data, n_context=self.hparams.n_context, passages_source_path=self.hparams.passages_source_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=self.collator
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, collate_fn=self.collator
        )

    def training_step(self, batch, batch_idx=None):
        # if self.training:
        #     return {
        #         "loss": torch.tensor(0.0, device=batch[0].device, requires_grad=True)
        #     }
        (index, target_ids, target_mask, passage_ids, passage_masks) = batch
        loss = self(input_ids=passage_ids, attention_mask=passage_masks, labels=target_ids)[0]
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    # TODO:完善Fid验证过程（不紧急）
    def validation_step(self, batch, batch_idx=None):
        (index, target_ids, target_mask, passage_ids, passage_masks) = batch
        loss = self(input_ids=passage_ids, attention_mask=passage_masks, labels=target_ids)[0]
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    def validation_epoch_end(self, outputs):
        acc = sum(outputs) / len(outputs)
        return acc

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.parameters(), self.hparams.learning_rate),
        ]
        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 200),
                "interval": "step",
            },
        ]

        return optimizers, schedulers


class FidQuestionAnsweringNQ(QuestionAnsweringNQ):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.net = FiDT5.from_pretrained(self.hparams.model_path)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.net.forward(input_ids, attention_mask, **kwargs)
    def generate(self, input_ids, attention_mask, max_length):
        return self.net.generate(input_ids, attention_mask, max_length)

def load_nq(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = load_all_jsonl(data_path)
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    logging.info("successfully load data, len {}".format(len(data)))
    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 do_softmax=False,
                 passages_source_path=None):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        # 是否对score softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        self.do_softmax = do_softmax
        # 不要进行排序，很多实验需要验证score的顺序影响
        if passages_source_path is not None:
            self.passages_source = pd.read_csv(passages_source_path, sep='\t')

    def get_text_from_id(self, id: int):
        # dataframe编码是从0开始的，id从1开始，所以index = id-1；可以输出例子确认一下
        return self.passages_source.at[id - 1, 'text'], self.passages_source.at[id - 1, 'title']

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            # python3以后删除了haskey方法
            # if not example['ctxs'][0].has_key('text'):
            if not 'text' in example['ctxs'][0]:
                for i in range(self.n_context):
                    # id字段默认是string类型，需要转换为int
                    # dpr_result id是以wiki:开头的
                    id = example['ctxs'][i]['id']
                    if id[0:5] == 'wiki:':
                        id = id.replace('wiki:', '')
                    id = int(id)
                    example['ctxs'][i]['text'], example['ctxs'][i]['title'] = self.get_text_from_id(id)

            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            if self.do_softmax:
                scores = self.softmax(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]

        else:
            passages, scores = None, None


        return {
            'index': index,
            'question': question,
            'target': target,
            'passages': passages,
            'scores': scores
        }

    def get_example(self, index):
        return self.data[index]
    
def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)
        ## passage包括question and passages
        return (index, target_ids, target_mask, passage_ids, passage_masks)
