#!/usr/bin/env python3
import pandas as pd
from collections import defaultdict
from CzcWav2vec2 import Wav2vec2_Gpt2
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    trainer_utils,
)
import time
import logging
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch

from datasets import load_dataset, load_from_disk
import os
os.environ["TRANSFORMERS_CACHE"] = "/data2_from_58175/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data2_from_58175/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/data2_from_58175/huggingface/metrics"
os.environ["HF_HOME"] = "/data2_from_58175/huggingface"
# os.environ["TMPDIR"] = "/data2/tmp"
logger = logging.getLogger(__name__)
def configure_logger(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if trainer_utils.is_main_process(training_args.local_rank):
        # 设置info只能在主进程显示，避免某些信息重复显示(多进程)
        logging_level = logging.INFO
    logger.setLevel(logging_level)
@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    datasets_name_or_path: str = field(
        metadata={"help": "Path to the datasets"}
    )
    total_jobs: Optional[int] = field(
        default=8,
        metadata={"help": "total devices for decoding, ie. num jobs"},
    )
    job_index: Optional[int] = field(
        default=1,
        metadata={"help": "total devices for decoding, ie. num jobs"},
    )
# time CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node 4 pseudo_decode_mp.py --model_name_or_path=/data2_from_58175/wav2vec2_output/filteredbyctc_continue/task1-50-rescore-checkpoint-2454-0.08439-0.08407-0.07993 --datasets_name_or_path=/home/data/fisher_swbd_nodup_onlyspeech/swbd_pseudo_89h --output_dir=/home/data/pseudo_decode_mp --total_devices=4
def show_args(args):
    print('\n'.join(['%s:%s' % item for item in args.__dict__.items()]))
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((Arguments, TrainingArguments))
    stars = "*"*20
    args, training_args = parser.parse_args_into_dataclasses()
    # info信息只在主进程中才显示，warning则是所有进程可用
    configure_logger(training_args)
    # parser = HfArgumentParser(Arguments)
    # parser.parse_args_into_dataclasses()输出默认是元组
    # args = parser.parse_args_into_dataclasses()
    total_jobs = args.total_jobs
    total_cudas = torch.cuda.device_count()
    logger.info(f"total_cudas = {total_cudas}")
    # kaldi/utils JOB索引只从1开始，故减1，从0开始
    job_index = args.job_index - 1
    cuda_index = job_index%total_cudas
    
    # manager = Manager()
    # my_list = manager.list()
    # my_list.append(cuda_index)
    # logger.warning(f"my_list = {my_list}")
    # cuda_index 若单卡则是-1，否则0,1,2,...N
    logger.info(f"args = {args}")
    device = "cuda:"+str(cuda_index)
    # device = "cpu"
    logger.warning(f"device = {device}")
    datasets_name_or_path = args.datasets_name_or_path

    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"{stars}loading dataset{stars}")

    dataset = load_from_disk(datasets_name_or_path)#.remove_columns(['sampling_rate', 'seg_end', 'seg_start'])
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))

    total_jobs = args.total_jobs
    total_examples = len(dataset)
    # total_examples = 200
    # 尽量等分，0到total_examples中total_devices等分，则有total_devices+1个点
    index_list = np.linspace(0,total_examples,total_jobs+1,dtype=int).tolist()
    start_index = index_list[job_index]
    end_index = index_list[job_index+1]
    logger.warning(f"start_index,end_index = {(start_index,end_index)}")
    text_ha = pd.read_csv("/data2_from_58175/fisher_swbd_nodup_script/id_text_ha_upper_")
    text_ha_uttidlist = text_ha["id"].tolist()
    def map2textha(batch):
        if batch["id"] in text_ha_uttidlist:
            text = text_ha[text_ha.id == batch["id"]].text.item()
            if text != batch["text"]:
                batch["text"] = text
        else:
            print(batch["id"])
        return batch
    result = dataset.select(range(start_index,end_index)).map(map2textha,keep_in_memory=True)
    logger.warning(f"{stars} over{stars}")
    logger.warning(f"start_index,end_index = {(start_index,end_index)}")

    subset_save_path = os.path.join(training_args.output_dir,str(job_index))
    logger.warning(f"subset is saved to path : {subset_save_path}")
    
    # ignored_columns = list(set(result.column_names) - set(['id', 'file', 'seg_start', 'seg_end', 'length', 'text']))
    # result = result.remove_columns(ignored_columns)
    logger.warning(f"finished subset :\n {result}")
    
    result.save_to_disk(subset_save_path)
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    
    info_idct = defaultdict(str)
    info_idct["model_name_or_path"] = args.model_name_or_path
    info_idct["datasets_name_or_path"] = args.datasets_name_or_path
    info_idct["total_jobs"] = args.total_jobs
    info_json = os.path.join(training_args.output_dir,"info.json")
    with open(info_json,"w",encoding="utf-8") as f:
        json.dump(info_idct,f,indent=2)
    # wer_rescore = wer(truth=result["text"],hypothesis=result["transcription"])
    # wer_bs = wer(truth=result["text"],hypothesis=result["transcription_bs"])
    # logger.info(f"model_path = {model_path}")
    # logger.info(f"datasets_name_or_path = {datasets_name_or_path}")
    # logger.info(f"wer_bs = {wer_bs}")
    # logger.info(f"wer_rescore = {wer_rescore}")

# jupyter中拼接
# subset_path_list = glob.glob("/home/data/pseudo_decode_mp/*")
# subset_path_list = sorted(subset_path_list)
# subset_list = []
# for i in subset_path_list:
    # subset = load_from_disk(i)
    # subset_list.append(subset)
# subset_list  
# swbd_simu_89h_ft1h_semi_50 = datasets.concatenate_datasets(subset_list)
# swbd_simu_89h_ft1h_semi_50.save_to_disk("/home/data/fisher_swbd_nodup_onlyspeech/swbd_pseudo_89h_ft1h_semi_50")

# /tsdata/kaldi_utils/run.pl JOB=1:30 /home/data/decode_mp/log/log.JOB.txt python decode_mp.py --model_name_or_path=/data2_from_58175/wav2vec2_output/filteredbyctc_continue/ft10h-ctcbs-0.08016-checkpoint-7320 --datasets_name_or_path=/home/data/fisher_swbd_nodup_onlyspeech/swbdtest5h --output_dir=/home/data/decode_mp --total_jobs=30 --job_index=JOB
if __name__ == "__main__":
    main()