#!/usr/bin/env python3
from speechbrain.lobes.augment import TimeDomainSpecAugment as sb_aug
from transformers import __version__
from beam_search_att_rescore import ctc_prefix_beam_search,attention_rescoring
from CzcWav2vec2 import Wav2vec2_Gpt2,Wav2Vec2ForCTC_semi
from pdb import set_trace
import torch.nn.functional as F
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
    AutoTokenizer,
    AutoModelForCausalLM,
    PretrainedConfig,
    BertConfig,
    EncoderDecoderConfig,
    GPT2Config,
    BertGenerationConfig
)
from transformers.modeling_outputs import Seq2SeqLMOutput
import time
import copy
from tqdm import tqdm
from itertools import groupby
from typing import Optional, Tuple
import math
import glob
from scipy import signal
import collections
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_utils import PreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import CzcWav2Vec2Model,CzcWav2Vec2ForCTC
from speechbrain.processing.speech_augmentation import SpeedPerturb
import logging
import pathlib
import re
import sys
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
import random
import torchaudio.sox_effects as sox_effects
import torchaudio as ta
import datasets
import numpy as np
import torch
import inspect
from packaging import version
from torch import nn

import librosa
from lang_trans import arabic

from datasets import load_dataset, load_from_disk
import os
os.environ["TRANSFORMERS_CACHE"] = "/data2_from_58175/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data2_from_58175/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/data2_from_58175/huggingface/metrics"
os.environ["HF_HOME"] = "/data2_from_58175/huggingface"
os.environ["TMPDIR"] = "/tsdata1/tmp"
if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


logger = logging.getLogger(__name__)

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(42, deterministic=False)

class Wav2Vec2Processor_Gpt_Tokenizer(Wav2Vec2Processor):
    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(feature_extractor, Wav2Vec2FeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {Wav2Vec2FeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, gpt_tokenizer_name_or_path, **kwargs):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_tokenizer_name_or_path, **kwargs)
        return cls(feature_extractor=feature_extractor, tokenizer=gpt_tokenizer)        
'''
class Wav2vec2_Gpt2(EncoderDecoderModel):
    def pad_list(self, xs: List[torch.Tensor], pad_value: int):
        """Perform padding for the list of tensors.

        Args:
            xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
            pad_value (float): Value for padding.

        Returns:
            Tensor: Padded tensor (B, Tmax, `*`).

        Examples:
            >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
            >>> x
            [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
            >>> pad_list(x, 0)
            tensor([[1., 1., 1., 1.],
                    [1., 1., 0., 0.],
                    [1., 0., 0., 0.]])

        """
        n_batch = len(xs)
        max_len = max([x.size(0) for x in xs])
        pad = torch.zeros(n_batch, max_len, dtype=xs[0].dtype, device=xs[0].device)
        pad = pad.fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]

        return pad


    def add_sos_eos(self, ys_pad: torch.Tensor, sos: int, eos: int,
                    ignore_id: int) -> torch.Tensor:
        """Add <sos> and <eos> labels.

        Args:
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
            sos (int): index of <sos>
            eos (int): index of <eeos>
            ignore_id (int): index of padding

        Returns:
            ys_in (torch.Tensor) : (B, Lmax + 1)
            ys_out (torch.Tensor) : (B, Lmax + 1)

        Examples:
            >>> sos_id = 10
            >>> eos_id = 11
            >>> ignore_id = -1
            >>> ys_pad
            tensor([[ 1,  2,  3,  4,  5],
                    [ 4,  5,  6, -1, -1],
                    [ 7,  8,  9, -1, -1]], dtype=torch.int32)
            >>> out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
            >>> ys_in
            tensor([[10,  1,  2,  3,  4,  5],
                    [10,  4,  5,  6, 11, 11],
                    [10,  7,  8,  9, 11, 11]])
            >>> ys_out
            tensor([[ 10, 1,  2,  3,  4,  5, 11],
                    [ 10, 4,  5,  6, 11, -1, -1],
                    [ 10, 7,  8,  9, 11, -1, -1]])
        """
        _sos = torch.tensor([sos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        _eos = torch.tensor([eos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    #     ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    #     ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
        ys_out = [torch.cat([_sos, y, _eos], dim=0) for y in ys]
        return self.pad_list(ys_out, ignore_id)
    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False  
    def forward(
        self,
        input_values=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        # self.encoder.wav2vec2输出的last_hidden_state与encoder_outputs.hidden_states[-1]一致
#         wav2vec2_hidden_state = self.encoder.wav2vec2(
#             input_values=input_values,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=False,
#             return_dict=return_dict,
#         )[0]     
#         print(wav2vec2_hidden_state)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_values=input_values,
                labels=labels,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

#         print(len(encoder_all_hidden_states))
        encoder_hidden_states = encoder_outputs.hidden_states[-1]
#         print(encoder_hidden_states.shape)

        
        # 传给encoder的labels不能直接传给decoder的input_ids及labels
        # 对于decoder_labels，对labels进行操作: labels中（非-100）首尾添加<bos>、<eos>
        # 对于decoder_input_ids，在decoder_labels基础上，需要用0替代-100，否则nn.embedding(-100)越界
        # 用0取代后pad的token不会对句子造成影响（attention_mask），一般输入token中没有0
        # 在decoder_labels基础上取dec_input_attention_mask
#         print(f"labels.shape={labels.shape}")
#         print(f"attention_mask.shape={attention_mask.shape}")        
        sos_id = self.decoder.config.bos_token_id
        eos_id = self.decoder.config.eos_token_id
#         print(f"sos_id={sos_id}")
#         print(f"eos_id={eos_id}")
        ignore_id = -100
        dec_labels = self.add_sos_eos(labels, sos_id, eos_id, ignore_id)
        dec_input_attention_mask = dec_labels.ne(-100)
        dec_input_ids = dec_labels.masked_fill(~dec_input_attention_mask, 0)
#         print(f"dec_labels.shape={dec_labels.shape}")
#         print(f"dec_input_attention_mask.shape={dec_input_attention_mask.shape}")
#         print(f"dec_input_ids.shape={dec_input_ids.shape}")
        # 不能用传入的attention_mask，那是音频采样点级别的，在cross_attention时需要帧级别的attention_mask
        with torch.no_grad():
            wav2vec2 = self.encoder.wav2vec2
            extract_features = wav2vec2.feature_extractor(input_values).transpose(1, 2)
            output_lengths = wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            enc_frame_attention_mask = torch.zeros(
                extract_features.shape[:2], dtype=extract_features.dtype, device=extract_features.device
            )
            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            enc_frame_attention_mask[
                (torch.arange(enc_frame_attention_mask.shape[0], device=extract_features.device), output_lengths - 1)
            ] = 1
            enc_frame_attention_mask = enc_frame_attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
#         print(f"enc_frame_attention_mask.shape:{enc_frame_attention_mask.shape}")
#         print(f"enc_frame_attention_mask:{enc_frame_attention_mask}")
        # Decode
        decoder_outputs = self.decoder(
            input_ids=dec_input_ids,
            attention_mask=dec_input_attention_mask ,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=enc_frame_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            labels=dec_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs
        # ctc_loss
        if encoder_outputs.loss.item() != 0:
            loss = (encoder_outputs.loss,decoder_outputs.loss)
        else:
            loss = (torch.tensor(0.0,device=encoder_outputs.loss.device),torch.tensor(0.0,device=decoder_outputs.loss.device))
        return Seq2SeqLMOutput(
            loss=loss,
            logits=(encoder_outputs.logits,decoder_outputs.logits),
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ) 
'''
class Wav2Vec2ForCTC_gpt(Wav2Vec2ForCTC):
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=50260,
                    reduction='sum',
                    zero_infinity=True,
                )
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    freeze_all_except_lm: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze all parameters of the model except lm_head."}
    )  
    freeze_ALN: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze parameters of freeze_feature_extractor and feed_forward."}
    ) 
    freeze_all_except_feature_extractor: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze all parameters of the model except feature_extractor."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient_checkpointing."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    use_gpt_tokenizer: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use_gpt_tokenizer,vocab size=50261."}
    )
    reinit_lm_head: Optional[bool] = field(
        default=False, metadata={"help": "Whether to reinitial lm_head"}
    )
    encoder_decoder_mode: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gpt2 as decoder"}
    )
    freeze_decoder: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze freeze parameters of decoder when in encoder_decoder_mode"}
    )
    pseudo_onthefly: Optional[bool] = field(
        default=False, metadata={"help": "Whether to update the pseudo_model or not,which create the pseudo_labels on the fly"}
    )
    freeze_w2v2forctc: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze freeze parameters of w2v2forctc when in encoder_decoder_mode"}
    )
    
def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    elif trainer_utils.is_main_process(training_args.local_rank):
        # 设置info只能在主进程显示，避免某些信息重复显示(多进程)
        logging_level = logging.INFO
    logger.setLevel(logging_level)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    target_text_column: Optional[str] = field(
        default="text",
        metadata={"help": "Column in the dataset that contains label (target text). Defaults to 'text'"},
    )
    speech_file_column: Optional[str] = field(
        default="file",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'file'"},
    )
    target_feature_extractor_sampling_rate: Optional[bool] = field(
        default=False,
        metadata={"help": "Resample loaded audio to target feature extractor's sampling rate or not."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "Filters out examples longer than specified. Defaults to no filtering."},
    )
    orthography: Optional[str] = field(
        default="librispeech",
        metadata={
            "help": "Orthography used for normalization and tokenization: 'librispeech' (default), 'timit', or 'buckwalter'."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    speed_perturb: Optional[bool] = field(
        default=False,
        metadata={"help": "apply speed perpturbation in collator."},
    )


@dataclass
class Orthography:
    """
    Orthography scheme used for text normalization and tokenization.

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to accept lowercase input and lowercase the output when decoding.
        vocab_file (:obj:`str`, `optional`):
            File containing the vocabulary.
        word_delimiter_token (:obj:`str`, `optional`, defaults to :obj:`"|"`):
            The token used for delimiting words; it needs to be in the vocabulary.
        translation_table (:obj:`Dict[str, str]`, `optional`, defaults to :obj:`{}`):
            Table to use with `str.translate()` when preprocessing text (e.g., "-" -> " ").
        words_to_remove (:obj:`Set[str]`, `optional`, defaults to :obj:`set()`):
            Words to remove when preprocessing text (e.g., "sil").
        untransliterator (:obj:`Callable[[str], str]`, `optional`):
            Function that untransliterates text back into native writing system.
    """

    do_lower_case: bool = False
    vocab_file: Optional[str] = None
    word_delimiter_token: Optional[str] = "|"
    translation_table: Optional[Dict[str, str]] = field(default_factory=dict)
    words_to_remove: Optional[Set[str]] = field(default_factory=set)
    untransliterator: Optional[Callable[[str], str]] = None

    @classmethod
    def from_name(cls, name: str):
        if name == "librispeech":
            return cls()
        if name == "timit":
            return cls(
                do_lower_case=True,
                # break compounds like "quarter-century-old" and replace pauses "--"
                translation_table=str.maketrans({"-": " "}),
            )
        if name == "buckwalter":
            translation_table = {
                "-": " ",  # sometimes used to represent pauses
                "^": "v",  # fixing "tha" in arabic_speech_corpus dataset
            }
            return cls(
                vocab_file=pathlib.Path(__file__).parent.joinpath("vocab/buckwalter.json"),
                word_delimiter_token="/",  # "|" is Arabic letter alef with madda above
                translation_table=str.maketrans(translation_table),
                words_to_remove={"sil"},  # fixing "sil" in arabic_speech_corpus dataset
                untransliterator=arabic.buckwalter.untransliterate,
            )
        raise ValueError(f"Unsupported orthography: '{name}'.")

    def preprocess_for_training(self, text: str) -> str:
        # TODO(elgeish) return a pipeline (e.g., from jiwer) instead? Or rely on branch predictor as is
        if len(self.translation_table) > 0:
            text = text.translate(self.translation_table)
        if len(self.words_to_remove) == 0:
            text = " ".join(text.split())  # clean up whitespaces
        else:
            text = " ".join(w for w in text.split() if w not in self.words_to_remove)  # and clean up whilespaces
        return text

    def create_processor(self, model_args: ModelArguments) -> Wav2Vec2Processor:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
        if self.vocab_file:
            tokenizer = Wav2Vec2CTCTokenizer(
                self.vocab_file,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                word_delimiter_token=self.word_delimiter_token,
            )
        else:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                word_delimiter_token=self.word_delimiter_token,
            )
        return Wav2Vec2Processor(feature_extractor, tokenizer)

class Add_Noise_Reverb(object):

    def __init__(self, musan_path, rir_path):
        
        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[40,50],'speech':[40,50],'music':[40,50]}
        # 三种类型噪声的采样数目，用于叠加多段，speech使用4-7段(模拟多人背景噪声)，noise或者music只使用一段
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        # noiselist的keys为 'noise'、'speech'、'music' 
        # values是对应的文件列表
        self.noiselist  = {}
        # musan_path="/tsdata/sre/musan"
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
        
        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
        # simulated_rirs_files有60000条
        # real_point_rirs_files有417条
        # pointsource_noises有843条
        # 总共有61260条混响样本
        self.simulated_rirs_files  = glob.glob(os.path.join(rir_path,'*/*/*/*.wav'))
        # 
        self.real_point_rirs_files = glob.glob(os.path.join(rir_path,'*/*.wav'))
#         self.rir_files = glob.glob(os.path.join("/tsdata/noise/RIRS_NOISES/real_rirs_isotropic_noises/",'*.wav'))
        self.rir_files  = self.simulated_rirs_files + self.real_point_rirs_files
        logger.info(f"noisetypes:{self.noisetypes}")
        for k,v in self.noiselist.items():
            logger.info(f"noisetype and num_files: {k,len(v)}")
        logger.info(f"num of rir_files:{len(self.rir_files)}")

    def additive_noise(self, noisecat, audio):
        # audio为numpy 2Darray
        # noisecat: {noise、speech、music}三选一

        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []
        # speech选择4-7段，noise或者music只选择一段
        audio_length = audio.shape[1]
        for noise in noiselist:
            noiseaudio, sample_rate = ta.load(noise)
            noiseaudio = noiseaudio.detach().numpy()
            noise_length = noiseaudio.shape[1]
            if noise_length <= audio_length:
                shortage    = audio_length - noise_length + 1
                # 若是填充一维数组 arr1D=np.array([1, 1, 2, 2, 3, 4])
                # np.pad(arr1D, (2, 3), 'wrap')代表首部补两个，尾部补三个
                # mode=warp决定用于填补的元素，warp是首尾相连型，得到[3, 4, 1, 1, 2, 2, 3, 4, 1, 1, 2]
                # 此例是二维数组，所以要补充(0,0)用于操作第一个维度，(0,0)即表示第一维度不变
                noiseaudio  = np.pad(noiseaudio, ((0,0),(0,shortage)), 'wrap')
                noiseaudio = noiseaudio[:,:audio_length]
            else:
                # 噪声样本长，则随机选与音频等长的段，不一定只取开头，这可以通过startframe的随机选择实现
                startframe = np.int64(random.random()*(noise_length-audio_length))
                noiseaudio = noiseaudio[:,int(startframe):int(startframe)+audio_length]
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        noise_sum = np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise_sum  + audio


    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        audio_length = audio.shape[1]
        rir,fs     = ta.load(rir_file)
        rir = rir.detach().numpy()
        rir_length = rir.shape[1]
        if rir_length <= audio_length:
            shortage    = rir_length - rir_length + 1
            rir       = np.pad(rir, ((0,0),(0, shortage)), 'wrap')
        else:
            startframe = np.int64(random.random()*(rir_length-audio_length))
            rir = rir[:,int(startframe):int(startframe)+audio_length]
        rir = rir / np.sqrt(np.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:audio_length]
    
    
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # print([feature["labels"][:1] for feature in features])
        input_features =[]
        # 统一值读取speech，节省硬盘空间（不用存储input_values）
        if "speech" in features[0].keys():
            # print("proccess speech")
            for feature in features:
                feature_normalized = (feature["speech"] - np.mean(feature["speech"])) / np.sqrt(np.var(feature["speech"]) + 1e-5)
                input_features.append({"input_values": feature_normalized})        
        # input_features = [{"input_values": feature["input_values"]} for feature in features]
        elif "input_values" in features[0].keys():
            # print("proccess input_values")
            input_features = [{"input_values": feature["input_values"]} for feature in features]
        else:
            raise ValueError(f"both ['input_values', 'speech'] not in {features} ")
        label_features = [{"input_ids": feature["labels"]} for feature in features]    
        # print([feature["input_values"][-10:] for feature in input_features])
        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # print(batch["input_values"].dtype)
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        # print(batch["input_values"][:,-5:])
        # print(batch["attention_mask"][:,-5:])
        # print(batch["labels"][:,-5:])
        return batch
@dataclass
class DataCollatorCTCWithPadding_Speed_Perturb:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    speeds = np.linspace(0.9,1.1,3).tolist()
    # sp_weights = (np.ones(3)).tolist()
#     sp_weights = (np.ones(21)*(1-0.3)/20).tolist()
#     sp_weights[10] = 0.5
#     print(sp_weights)
#     speeds = [0.9,1.0,1.1]
    sp_weights = [1,1,1]
    logger.info("{stars}doing speed perturbation in collect_func{stars}") if len(speeds)>1 else None
    add_no_re = False
    logger.info("{stars}adding noise in collect_func{stars}") if add_no_re else None
    add_noise_reverb = Add_Noise_Reverb(musan_path="/tsdata/sre/musan", rir_path="/tsdata/sre/RIRS_NOISES/") if add_no_re else None
    # pitchs = np.linspace(-100,100,21).tolist()
    # pit_weights = np.ones(21).tolist()
    # volumns = np.linspace(0.12,2,81).tolist()
    # vol_weights = np.ones(81).tolist()
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        
        # print([feature["labels"][:1] for feature in features])
        speed = random.choices(self.speeds, self.sp_weights, k=1)[0]
        # volumn = random.choices(self.volumns, self.vol_weights, k=1)[0]
        # pitch = random.choices(self.pitchs, self.pit_weights, k=1)[0]
        if "speech" in features[0].keys():
            input_features = []
            # 终版，问题出在input_values已经是normalize后的，而torchaudio处理原始信号比较精准，所以我们采用speech作为处理信号，而后再归一化
            # 可以用第二个版本
            for feature in features:
                if speed != 1.0:
                # torchaudio需要二维数组(1,*)，输出也是二维数组，所以再调用时后输出后需要unsqueeze(0)和squeeze(0)
                    feature_speed_perturbed = sox_effects.apply_effects_tensor(
                        tensor = torch.tensor(feature["speech"]).unsqueeze(0),
                        sample_rate = 16000,
                        effects = [['speed', str(speed)], ['rate', str(16000)]]
                    )[0]
                else:
                    feature_speed_perturbed = torch.tensor(feature["speech"]).unsqueeze(0)
                if self.add_no_re and speed==1.0:
                    # 1-4则加混响或者加噪
                    augtype = random.choices([2,3,4], [1,1,1], k=1)[0]
                    feature_speed_perturbed_np = feature_speed_perturbed.numpy()
                    # 输入为2D array
                    if augtype == 1:
                        audio_aug = self.add_noise_reverb.reverberate(feature_speed_perturbed_np)[0]
                    elif augtype == 2:
                        audio_aug = self.add_noise_reverb.additive_noise('music',feature_speed_perturbed_np)[0]
                    elif augtype == 3:
                        audio_aug = self.add_noise_reverb.additive_noise('speech',feature_speed_perturbed_np)[0]
                    elif augtype == 4:
                        audio_aug = self.add_noise_reverb.additive_noise('noise',feature_speed_perturbed_np)[0]
                    else:
                        audio_aug = feature_speed_perturbed_np[0]  
                    # audio_aug : 1D array            
                    audio_aug = (audio_aug - np.mean(audio_aug)) / np.sqrt(np.var(audio_aug) + 1e-5)            
                else:
                    feature_speed_perturbed = feature_speed_perturbed[0]
                    audio_aug = (feature_speed_perturbed - torch.mean(feature_speed_perturbed)) / torch.sqrt(torch.var(feature_speed_perturbed) + 1e-5)
                input_features.append({"input_values": audio_aug})
        else:
            raise ValueError(f" ['speech'] is required for speech perpturbation ")

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # print(batch["input_values"].dtype) 
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        # print(batch["input_values"][:,-5:])
        # print(batch["attention_mask"][:,-5:])
        # print(batch["labels"][:,-5:])
        return batch
        
@dataclass
class DataCollatorCTCWithPadding_Speed_Perturb_sb:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    add_no_re = False
    logger.info("adding noise in collect_func") if add_no_re else None
    add_noise_reverb = Add_Noise_Reverb(musan_path="/tsdata/sre/musan", rir_path="/tsdata/sre/RIRS_NOISES/") if add_no_re else None
    feature_maker = sb_aug(
                perturb_prob=1.0,
                drop_freq_prob=0.0,
                drop_chunk_prob=0.0,
                speeds=[95, 100, 105],           
                )
    # pitchs = np.linspace(-100,100,21).tolist()
    # pit_weights = np.ones(21).tolist()
    # volumns = np.linspace(0.12,2,81).tolist()
    # vol_weights = np.ones(81).tolist()
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if "speech" in features[0].keys():
            input_features = []
            # 终版，问题出在input_values已经是normalize后的，而torchaudio处理原始信号比较精准，所以我们采用speech作为处理信号，而后再归一化
            # 可以用第二个版本
            for feature in features:
                speech_2d_tensor = torch.tensor(feature["speech"]).unsqueeze(0)
                feature_speed_perturbed = self.feature_maker(speech_2d_tensor,torch.ones(1))[0]
                audio_aug = (feature_speed_perturbed - torch.mean(feature_speed_perturbed)) / torch.sqrt(torch.var(feature_speed_perturbed) + 1e-5)
                input_features.append({"input_values": audio_aug})
        else:
            raise ValueError(f" ['speech'] is required for speech perpturbation ")

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # print(batch["input_values"].dtype) 
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        # print(batch["input_values"][:,-5:])
        # print(batch["attention_mask"][:,-5:])
        # print(batch["labels"][:,-5:])
        return batch
        
# 做变速需要设置input_column_change，全局变量的设置依赖于data_args.speed_perturb，所以将参数获取从main中抽取出来
# parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
# model_args, data_args, training_args = parser.parse_args_into_dataclasses()
# input_column_change = True if data_args.speed_perturb else False
class CTCTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        train_dataset_teacher: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        processor: Optional[Wav2Vec2Processor] = None,
        pseudo_model: Union[PreTrainedModel, nn.Module] = None,
        data_collator_eval: Optional[DataCollator] = None,
        teacher_model: Union[PreTrainedModel, nn.Module] = None,
        encoder_decoder_mode: Optional[bool] = None,
    ):
        super(CTCTrainer,self,).__init__(
                                model,
                                args,
                                data_collator,
                                train_dataset,
                                eval_dataset,
                                tokenizer,
                                model_init,
                                compute_metrics,
                                callbacks,
                                optimizers,                                
                                )
        self.processor = processor
        self.pseudo_model = pseudo_model
        self.data_collator_eval = data_collator_eval
        self.teacher_model = teacher_model
        self.encoder_decoder_mode = encoder_decoder_mode
        ##############
        self.train_dataset_teacher = train_dataset_teacher
        self.best_wer = 1.0
        self.update_pseudo_model = False
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        global input_column_change
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids"]
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        
        # 同一保留speech，可用于做变速，并重新做normalization，即为input_values
        ignored_columns.remove("speech") if "speech" in ignored_columns else None
        # if input_column_change:
            # ignored_columns.remove("speech")
            # ignored_columns.append("input_values")
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        # if flag:
            # torch.save(inputs, "./input_values_.pt")
            # flag = False
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.args.device)
                if self.deepspeed and inputs[k].dtype != torch.int64:
                    # NLP models inputs are int64 and those get adjusted to the right dtype of the
                    # embedding. Other models such as wav2vec2's inputs are already float and thus
                    # may need special handling to match the dtypes of the model
                    kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))

                inputs[k] = v.to(**kwargs)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs
    def _get_train_sampler_teacher(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_dataset_teacher, collections.abc.Sized):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset_teacher, datasets.Dataset):
                lengths = (
                    self.train_dataset_teacher[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset_teacher.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset_teacher,
                    self.args.train_batch_size,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset_teacher,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(self.train_dataset_teacher, generator=generator)
                return RandomSampler(self.train_dataset_teacher)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset_teacher,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset_teacher,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
    def get_train_dataloader_teacher(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset_teacher is None:
            raise ValueError("Trainer: training requires a train_dataset_teacher.")

        train_dataset_teacher = self.train_dataset_teacher
        if is_datasets_available() and isinstance(train_dataset_teacher, datasets.Dataset):
            train_dataset_teacher = self._remove_unused_columns(train_dataset_teacher, description="training")

        if isinstance(train_dataset_teacher, torch.utils.data.dataset.IterableDataset):
            if self.args.world_size > 1:
                train_dataset_teacher = IterableDatasetShard(
                    train_dataset_teacher,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset_teacher,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler_teacher()

        return DataLoader(
            train_dataset_teacher,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.dataset.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator_eval,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)
        logger.info("using data_collator_eval with no augmentation") 
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator_eval,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    # def prediction_step_mtl(
        # self,
        # model: nn.Module,
        # inputs: Dict[str, Union[torch.Tensor, Any]],
        # prediction_loss_only: bool,
        # ignore_keys: Optional[List[str]] = None,
    # ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # """
        # Perform an evaluation step on :obj:`model` using obj:`inputs`.

        # Subclass and override to inject custom behavior.

        # Args:
            # model (:obj:`nn.Module`):
                # The model to evaluate.
            # inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                # The inputs and targets of the model.

                # The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                # argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            # prediction_loss_only (:obj:`bool`):
                # Whether or not to return the loss only.
            # ignore_keys (:obj:`Lst[str]`, `optional`):
                # A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                # gathering predictions.

        # Return:
            # Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            # logits and labels (each being optional).
        # """
        # has_labels = all(inputs.get(k) is not None for k in self.label_names)
        # inputs = self._prepare_inputs(inputs)
        # if ignore_keys is None:
            # if hasattr(self.model, "config"):
                # ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            # else:
                # ignore_keys = []

        # # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        # if has_labels:
            # labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            # if len(labels) == 1:
                # labels = labels[0]
        # else:
            # labels = None

        # with torch.no_grad():
            # if is_sagemaker_mp_enabled():
                # raw_outputs = smp_forward_only(model, inputs)
                # if has_labels:
                    # if isinstance(raw_outputs, dict):
                        # loss_mb = raw_outputs["loss"]
                        # logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    # else:
                        # loss_mb = raw_outputs[0]
                        # logits_mb = raw_outputs[1:]

                    # loss = loss_mb.reduce_mean().detach().cpu()
                    # logits = smp_nested_concat(logits_mb)
                # else:
                    # loss = None
                    # if isinstance(raw_outputs, dict):
                        # logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    # else:
                        # logits_mb = raw_outputs
                    # logits = smp_nested_concat(logits_mb)
            # else:
                # if has_labels:
                    # ctc_weight = 0.2
                    # loss_tuple, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    # ctc_loss,att_loss = loss_tuple
                    # loss = ctc_weight * ctc_loss + (1-ctc_weight) * att_loss
                    # loss = loss.mean().detach()
                    # if isinstance(outputs, dict):
                        # logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    # else:
                        # logits = outputs[1:]
                # else:
                    # loss = None
                    # if self.use_amp:
                        # with autocast():
                            # outputs = model(**inputs)
                    # else:
                        # outputs = model(**inputs)
                    # if isinstance(outputs, dict):
                        # logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    # else:
                        # logits = outputs
                    # # TODO: this needs to be fixed and made cleaner later.
                    # if self.args.past_index >= 0:
                        # self._past = outputs[self.args.past_index - 1]

        # if prediction_loss_only:
            # return (loss, None, None)

        # logits = nested_detach(logits)
        # if len(logits) == 1:
            # logits = logits[0]

        # return (loss, logits, labels)
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            if self.encoder_decoder_mode:
                # loss, logits, labels = self.prediction_step_mtl(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                loss, logits, labels = self.prediction_step(model.encoder, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            else:
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
            self.update_pseudo_model = True if metrics["wer"] < self.best_wer else False
            self.best_wer = metrics["wer"] if metrics["wer"] < self.best_wer else self.best_wer
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, loss_tuple):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            ctc_loss, additional_loss = loss_tuple[0].detach(),loss_tuple[1].detach()
            ctc_loss_scalar = ctc_loss.item()
            additional_loss_scalar = additional_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss
            ctc_loss -= ctc_loss
            additional_loss -= additional_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
#########################################################
            if additional_loss_scalar == 0:
                # 处于纯ctc模式
                logs["ctc_loss"] = logs["loss"]
            else:
                logs["ctc_loss"] = round(ctc_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["additional_loss"] = round(additional_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._total_ctc_loss_scalar += ctc_loss_scalar
            self._total_additional_loss_scalar += additional_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self.model = self.model.to(args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # 是否进行交替时训练，混合数据训练1个epoch，再单独用标注数据训练1个epoch，如此往复
        train_crossover = False
        # Data loader and number of training steps
        # train_dataloader = self.get_train_dataloader()
        train_dataloader_mix = self.get_train_dataloader()
        if train_crossover == True:
            # 单独的真实标签数据
            train_dataloader_teacher = self.get_train_dataloader_teacher()
            train_dataloader = None
        else:
            train_dataloader_teacher = None
            train_dataloader = train_dataloader_mix
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        # logger.info(f"args.max_steps = {args.max_steps}")
        # args.max_steps = -1
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            # 
            #############################
            teacher_inter_epochs = 5
            # 表示每几个epoch后进行纯标注数据训练
            if train_crossover == True:
                # 实际上是两个epochs
                num_update_steps_per_epoch = (teacher_inter_epochs * len(train_dataloader_mix) + len(train_dataloader_teacher))// args.gradient_accumulation_steps
            else:
                num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                
                # 10epochs 其中5epochs用混合数据，5epochs用仅带标注数据, num_update_steps_per_epoch代表是交替一轮
                if train_crossover == True:
                    max_steps = math.ceil(args.num_train_epochs / (teacher_inter_epochs+1) * num_update_steps_per_epoch)
                else:
                    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                ######################
                
                if train_crossover == True:
                    num_train_samples = (teacher_inter_epochs * len(self.train_dataset) + len(self.train_dataset_teacher)) * args.num_train_epochs
                else:
                    num_train_samples = len(self.train_dataset) * args.num_train_epochs
                # num_train_samples是num_examples的epochs倍
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            num_train_epochs = int(args.num_train_epochs)
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!

        ####################
        if train_crossover == True:
            num_examples = (
                (self.num_examples(train_dataloader_mix) + self.num_examples(train_dataloader_teacher)) if train_dataset_is_sized else total_train_batch_size * args.max_steps
            )
        else:
            num_examples = (
                self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
            )
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        #####################
        if train_crossover == True:
            self.callback_handler.train_dataloader = train_dataloader_mix
        else:
            self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
#########################################################
        ctc_loss = torch.tensor(0.0).to(args.device)
        additional_loss = torch.tensor(0.0).to(args.device)
        self._total_ctc_loss_scalar = 0.0
        self._total_additional_loss_scalar =0.0
        
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses        
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader_mix:
                    break
#################################初始2个dict保存最后两个epoch的结果，deepcopy才能将dict完全与模型脱离开，否则模型更新时二者会同步
        if self.pseudo_model is not None:
            # print(self.pseudo_model)
            # self.pseudo_model.to(model.device)
            logger.info("initialing state_dict_pseudo and pseudo_model")
            state_dict_pseudo = copy.deepcopy(self.pseudo_model.state_dict())
            # state_dict_pseudo = torch.load("/data2_from_58175/wav2vec2_output/fisher_joint/-checkpoint-5000/pytorch_model.bin",map_location="cpu")
            # print(state_dict_pseudo)
            self.pseudo_model.load_state_dict(state_dict_pseudo)
            
            # encoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/encoder"
            # decoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/decoder"
            # encoder__ = Wav2Vec2ForCTC.from_pretrained(encoder_model_path)
            # decoder__ = AutoModelForCausalLM.from_pretrained(decoder_model_path)
            # self.pseudo_model_ft = Wav2vec2_Gpt2(encoder=encoder__,decoder=decoder__)
            # state_dict_ft = torch.load("/home/wav2vec2_output/fisher_swbd_ft_1h_lsm_wd3_ctc5/-checkpoint-420/pytorch_model.bin",map_location="cpu")
            # self.pseudo_model_ft.load_state_dict(state_dict_ft)
            # self.pseudo_model.load_state_dict(state_dict_ft)
            self.pseudo_model.to(model.device)
            
            # pseudo_model = self._wrap_model(pseudo_model, training=False)
            # 设置更新节点，每10%更新一次
            onthefly_point = [int(max_steps*i) for i in np.arange(3)/2]
            logger.info(f"we will update the pseudo_model at {onthefly_point[1:]} step")
            # 某节点在更新后，其value置为True，防止某节点多次重复更新
            onthefly_update_state = {i:False for i in onthefly_point}
            # key代表更新伪标签的数量（batch size=5），为True代表已到该阶段并输出到屏幕过
            num_pl_update_dict = {i:False for i in [0,1,2,3,4,5]}
            # onthefly_update_state = {i:False for i in onthefly_point}
            # 由于state_dict从model复制来，而model原是train模式，存在随机dropout，生成的伪标签质量不可靠，训练崩溃，eval()不能省
            self.pseudo_model.eval()
        # 是否使用pseudo模型生成软标签计算loss
        self.decoder_soft_loss = False
        self.encoder_soft_loss = False
        self.do_rdop = False
        self.decode_bs_rescoring = False
        for epoch in range(epochs_trained, num_train_epochs):
            self.update_pseudo_model = False 
            # if epoch==0:
                # print("only finetune the parameters of feature_extractor in 1st epoch")
                # model.module.freeze_all_except_feature_extractor()
            # else:
                # # 训练过程中不允许再变动param.requires_grad
                # print("freeze the parameters of feature_extractor from now")
                # model.module.classical_freeze()
            if train_crossover == True:
                if epoch % teacher_inter_epochs == 0 and epoch != 0:
                    logger.info("train_dataloader = train_dataloader_teacher")
                    train_dataloader = train_dataloader_teacher
                else:
                    logger.info("train_dataloader = train_dataloader_mix")
                    train_dataloader = train_dataloader_mix
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            for step, inputs in enumerate(epoch_iterator) or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
#################################最初点不进行更新,直接使用state_dict_pseudo        
                if (self.pseudo_model is not None):
                    if self.update_pseudo_model:
                        # 会循环执行gradient_accumulation_steps次，而我们只需要更新一次模型
                        # 例如step=0,1,2,3,4,5,6,7对应的global_steps都相同，我们只在step=0更新pseudo_model
                        # 假设为onthefly_point为[0, 7, 14, 21, 28, 35, 42, 49, 56, 63]时更新，而gradac_step=8，在max_step=14时进入更新循环的step可能有两个是8的倍数，例如112,0，因为一个epoch已经结束
                        # 设置onthefly_update_state解决此问题
                        if step% args.gradient_accumulation_steps == 0 and (self.state.global_step == onthefly_point[0]):
                            if onthefly_update_state[self.state.global_step]==False:
                                # print(step)
                                # print(f"update pseudo_model at {onthefly_point} step")
                                logger.info(f"current step = {self.state.global_step},updating the pseudo_model")
                                # print("before:",model.module.encoder.lm_head.weight)
                                state_dict = copy.deepcopy(model.module.state_dict())
                                # 同时赋值才能保证state_dict_last不会等于state_dict_last_last
                                # print(state_dict_pseudo!=state_dict_last)
                                # assert (state_dict_pseudo!=state_dict_last),f"state_dict_last is equal to state_dict_pseudo"
                                # pseudo_model_type = random.choices([1,2,3], [1,0,0], k=1)[0]
                                # if pseudo_model_type == 1:
                                    # print("pseudo_model = pseudo_model")
                                    # state_dict = state_dict_pseudo
                                # elif pseudo_model_type == 2:
                                    # print("pseudo_model = last_epoch_model")
                                    # state_dict = state_dict_last
                                # else:
                                    # print("pseudo_model = 0.5*last_epoch_model + 0.5*pseudo_model")
                                    # state_dict = state_dict_last
                                    # for k in state_dict.keys():
                                        # state_dict[k] = 0.5*state_dict[k] + 0.5*state_dict_pseudo[k]
                                online_model_weight = 0.5
                                # online_model_weight = self.state.global_step/max_steps/2
                                logger.info(f"online_model_weight = {online_model_weight}")
                                for k in state_dict.keys():
                                    state_dict[k] = online_model_weight*state_dict[k] + (1-online_model_weight)*state_dict_pseudo[k]
                                self.pseudo_model.load_state_dict(state_dict)
                                # print("after:",model.module.encoder.lm_head.weight)
                                onthefly_update_state[self.state.global_step]=True
                    # else:
                        # self.pseudo_model.load_state_dict(state_dict_pseudo)
                        
                # Skip past any already trained steps if resuming training
                # print(inputs["input_values"][:,-5:])
                # print(inputs["attention_mask"][:,-5:])
                # print(inputs["labels"][:,-5:])
#                 set_trace()
                if self.do_rdop == True:
                    inputs = self._prepare_inputs(inputs)
                    with torch.no_grad():
                        self.encoder_teacher_logits,self.decoder_teacher_logits = model(**inputs).logits
                if self.pseudo_model is not None:# and self.update_pseudo_model: #and (self.state.global_step >= onthefly_point[1]) 
                    
                    # pseudo_model_choice = random.choices([0,1], [1,1], k=1)[0]
                    # # logger.info(f"pseudo_model_choice = {pseudo_model_choice}")
                    # self.pseudo_model = self.pseudo_model if pseudo_model_choice==0 else self.pseudo_model_ft
                    # self.pseudo_model.to(model.device)
                    
                    self.pseudo_model.eval()
                    inputs = self._prepare_inputs(inputs)
                    # 不动态更新伪标签，则注释下块
                    # tmp = inputs["labels"]
                    # print(inputs["input_values"])
                    

                    if self.decode_bs_rescoring:
                        # assert inputs["labels"].shape[0] == 1
                        pseudo_ids = []
                        for i in range(inputs["labels"].shape[0]):
                            (ctc_beam_score,att_score,rescore),pseudo_id = attention_rescoring(input_values=inputs["input_values"][i].unsqueeze(0),
                                                                                 attention_mask=inputs["attention_mask"][i].unsqueeze(0),
                                                                                 model=self.pseudo_model,
                                                                                 beam_size=4,
                                                                                 ctc_weight=0.5,
                                                                                 output_prefix_beam_search=False)
                            # logger.info(f"pseudo_id = {pseudo_id}")
                            pseudo_ids.append(pseudo_id[0])
                        # logger.info(f"pseudo_ids = {pseudo_ids}")
                        # bs_rescoring得到的就是labels，并不是帧级别输出
                        labels_onthefly = pseudo_ids
                        # pseudo_trans = self.processor.batch_decode(pseudo_ids,group_tokens=False)
                    else:
                        if isinstance(self.pseudo_model,Wav2Vec2ForCTC):
                            with torch.no_grad():
                                self.encoder_teacher_logits = self.pseudo_model(input_values=inputs["input_values"].to(self.pseudo_model.device),attention_mask=inputs["attention_mask"].to(self.pseudo_model.device)).logits
                            pseudo_ids = torch.max(self.encoder_teacher_logits , dim=-1)[1]
                        elif isinstance(self.pseudo_model,Wav2vec2_Gpt2):
                            # ctc_prefix_beam_search只支持bs=1
                            # pseudo_ids = attention_rescoring(input_values=inputs["input_values"].to(self.pseudo_model.device),
                                                                                 # attention_mask=inputs["attention_mask"].to(self.pseudo_model.device),
                                                                                 # model=self.pseudo_model,
                                                                                 # beam_size=4,
                                                                                 # ctc_weight=0.5,
                                                                                 # output_prefix_beam_search=False)
                            with torch.no_grad():
                                self.encoder_teacher_logits = self.pseudo_model.encoder(input_values=inputs["input_values"].to(self.pseudo_model.device),attention_mask=inputs["attention_mask"].to(self.pseudo_model.device)).logits
                                # self.encoder_teacher_logits,self.decoder_teacher_logits = self.pseudo_model(**inputs).logits
                            pseudo_ids = torch.max(self.encoder_teacher_logits , dim=-1)[1]
                            pseudo_trans = self.processor.batch_decode(pseudo_ids)
                            # 个别样本是静音段但是带标签，导致trans=""，processor.as_target_processor()无法处理空字符串，可将""转为"|"
                            # 如果是转成" "，当整个batch都为[' ', ' ', ' ', ' ', ' ', ' ']同样会出错
                            pseudo_trans = ["|" if i=="" else i for i in pseudo_trans]
                            # logger.info(f"pseudo_trans = {pseudo_trans}")
                            # print(f"pseudo_rows = {pseudo_rows}")
                            with self.processor.as_target_processor():
                                labels_onthefly = self.processor(pseudo_trans).input_ids  
                    pseudo_mask = (inputs["labels"]==-1)
                    pseudo_rows = pseudo_mask.nonzero()[:,0].tolist()
                    # print(f"labels_onthefly:{labels_onthefly}")
                    labels = inputs["labels"].tolist()
                    # 将labels中-1所在行的伪标签id换成模型此刻预测的token_id
                    
                    # 不全部实时更新，而是按进度条百分比更新
                    # 第0个阶段更新0个伪标签，第i个阶段更新i个伪标签(1<i<=batch_size)
                    
                    # if self.state.global_step >= onthefly_point[1:]:
                        # num_pl_update = onthefly_point.index(self.state.global_step)
                        # num_pl_update = min(2,num_pl_update,len(pseudo_rows))
                    # num_pl_update = min(2,len(pseudo_rows))
                    # if num_pl_update_dict[num_pl_update] == False:
                        # print(f"num_pl_update = {num_pl_update}")
                        # num_pl_update_dict[num_pl_update] = True
                        
                    # for i in random.sample(pseudo_rows,num_pl_update):
                    for i in pseudo_rows:
                        # 仍然在伪标签前加上"-1"标志
                        labels[i] = [-1]+labels_onthefly[i]
                    # # 新labels需要重新pad
                    # # print(f"tmp= {tmp.shape}")
                    # # print(f"pseudo_labels = {inputs['labels'].shape}")
                    inputs["labels"] = pad_sequence([torch.from_numpy(np.array(i)) for i in labels],batch_first=True,padding_value=-100).to(model.device)
                    # logger.info(f"inputs['labels'] = {inputs['labels']}")
                    # 标签更新后计算decoder_teacher_logits, encoder_teacher_logits则不需要标签
                    if self.decoder_soft_loss:
                        # self.pseudo_model.train()
                        with torch.no_grad():
                            _,self.decoder_teacher_logits = self.pseudo_model(**inputs).logits
                    # inputs["input_values"].to(model.device)
                    # inputs["attention_mask"].to(model.device)
                    # print(f"pseudo_labels = {tmp[:,:20]}")
                    # print(f"pseudo_labels shape = {tmp.shape}")
                    # print(f"onethefly labels = {inputs['labels'][:,:20]}")
                    # print(f"onethefly labels.shape:{inputs['labels'].shape}") 
                    # print((tmp!=inputs['labels']).sum()) if tmp.shape == inputs['labels'].shape else print("shape is not the same")
                # print(f"pseudo_labels = {inputs['labels']}")
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
#########################################################
                    with model.no_sync():   
                        if not self.encoder_decoder_mode:
                            # set_trace()
                            tr_loss_ = self.training_step(model, inputs)
                            tr_loss += tr_loss_
                            loss_tuple = torch.tensor(0),torch.tensor(0)
                        else:
                            tr_loss_,loss_tuple_ = self.training_step(model, inputs)
                            ctc_loss_, additional_loss_ = loss_tuple_
                            # print(tr_loss,ctc_loss,additional_loss)
                            tr_loss += tr_loss_
                            ctc_loss += ctc_loss_.detach()
                            additional_loss += additional_loss_.detach()
                            loss_tuple = (ctc_loss,additional_loss)
                else:
#                     print(self.teacher_model)
#########################################################
                    if not self.encoder_decoder_mode:
                        # set_trace()
                        tr_loss_ = self.training_step(model, inputs)
                        tr_loss += tr_loss_
                        loss_tuple = torch.tensor(0),torch.tensor(0)
                    else:
                        tr_loss_,loss_tuple_ = self.training_step(model, inputs)
                        tr_loss += tr_loss_
                        ctc_loss_, additional_loss_ = loss_tuple_
                        ctc_loss += ctc_loss_.detach()
                        additional_loss += additional_loss_.detach()
                        loss_tuple = (ctc_loss,additional_loss)
                # print(f"tr_loss={tr_loss}")
                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()
                
                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, loss_tuple)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, loss_tuple)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME), map_location="cpu")
            # If the model is on the GPU, it still works!
            self._load_state_dict_in_model(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.pseudo_model is not None and self.decoder_soft_loss and self.model.training:
            # 在eval(self.model.training=Fasle)时不需要传入teacher_logits
            # with torch.no_grad():
                # self.encoder_teacher_logits,self.decoder_teacher_logits = self.pseudo_model(**inputs).logits
                # print(f"self.decoder_teacher_logits.shape = {self.decoder_teacher_logits.shape}")
            if not self.encoder_soft_loss:
                outputs = model(**inputs,decoder_teacher_logits=self.decoder_teacher_logits)
            else:
                outputs = model(**inputs,encoder_teacher_logits=self.encoder_teacher_logits,decoder_teacher_logits=self.decoder_teacher_logits)
        elif self.do_rdop and self.model.training:
            outputs = model(**inputs)
            encoder_logits,decoder_logits = outputs.logits
            rdop_encoder_loss = self.compute_kl_loss(encoder_logits,self.encoder_teacher_logits) * 0.01
            rdop_decoder_loss = self.compute_kl_loss(decoder_logits,self.decoder_teacher_logits) * 0.1
            # print(f"rdop_encoder_loss = {rdop_encoder_loss}")
            # print(f"rdop_decoder_loss = {rdop_decoder_loss}")
            # print(f"outputs['loss'] = {outputs['loss']}")
            encoder_loss = outputs['loss'][0] + rdop_encoder_loss
            decoder_loss = outputs['loss'][1] + rdop_decoder_loss
            outputs['loss'] = (encoder_loss, decoder_loss)
            # print(f"outputs['loss'] = {outputs['loss']}")
        else:
            outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(f"sigle gpu loss:{loss}")
        return (loss, outputs) if return_outputs else loss
    def compute_kl_loss(self,p,q):
        q_loss = F.kl_div(q.log_softmax(dim=-1), p.softmax(dim=-1), reduction='sum')
        p_loss = F.kl_div(p.log_softmax(dim=-1), q.softmax(dim=-1), reduction='sum')
        return (q_loss+p_loss)/2  
    def compute_distill_loss(self, model, teacher_model, inputs, return_outputs=False):
        """
        计算帧级别相对熵损失，例如[[[0.6,0.4],[0.3,0.7]]]，制作其标签为[[[1,0],[0,1]]]
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # num_char_tokens = (inputs["labels"] >= 0).sum()
        # print(f"num_char_tokens={num_char_tokens}")
        outputs = model(**inputs)
        logits = outputs["logits"]
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs_logits = model.to(logits.device)(**inputs)["logits"]
#         log_probs_mat = logits
        # print(f"log_probs_mat.shape={log_probs_mat.shape}")
#         log_probs_mat_mask = log_probs_mat.argmax(dim=-1,keepdim=True)
#         log_probs_mat_onehot = torch.zeros(log_probs_mat.shape,device=log_probs_mat.device).scatter_(-1, log_probs_mat_mask, 1)  
        # print(f"log_probs_mat_onehot.shape={log_probs_mat_onehot.shape}")
        # print(f"log_probs_mat_mask.shape={log_probs_mat_mask.shape}")
        distill_loss = self.compute_kl_loss(logits,teacher_outputs_logits)
        # kl_loss = self.compute_kl_loss(log_probs_mat,log_probs_mat_onehot)/num_char_tokens           
#         distill_weight = 0.001
        
        # Save past state if it exist
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"]
#             print(f"ctc_loss={loss}")
#             print(f"distill_loss.requires_grad={distill_loss.requires_grad}")
            loss = (loss,distill_loss)
#             print(f"distill_weight*distill_loss={distill_weight*distill_loss}")
            # print(f"total_loss={loss}")
        return (loss, outputs) if return_outputs else loss     
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        # global flag
        model.train()
        inputs = self._prepare_inputs(inputs)
        # print(inputs["input_values"][0,-5:])
        # print(inputs["attention_mask"][0,-5:])
        # print(inputs["labels"][0,-5:])          
        
        # if flag:
            # torch.save(inputs, "./input_values_.pt")
            # flag = False
        if self.use_amp:
            with autocast():
#########################################################
                if self.encoder_decoder_mode:
                    # training_step DDP 各卡算各自的，每批都会进行loss的反向传播并计算梯度
                    ctc_loss,att_loss = self.compute_loss(model, inputs)  
                    # print(ctc_loss)
#                         print(f"sum ctc_loss",{ctc_loss.item()} )
#                         print(f"sum_att_loss={att_loss.item()}")
                else:
                    loss = self.compute_loss(model, inputs)  
                    # print(loss)

            # print(f"loss with autocast={loss}")
        else:
            loss = self.compute_loss(model, inputs)
            # print(f"loss without autocast={loss}")
        # 有点问题，DDP下self.args.n_gpu被设置为1，则不进入if
        # if self.args.n_gpu > 1:
            # if model.module.config.ctc_loss_reduction == "mean":
                # loss = loss.mean()
            # elif model.module.config.ctc_loss_reduction == "sum":
                # loss = loss.sum() / (inputs["labels"] >= 0).sum()
            # else:
                # raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
        freeze_decoder = False
        ctc_weight = 0.2
        if self.encoder_decoder_mode and freeze_decoder:
            ctc_weight = 1.0
        if model is not self.model :
            # print("####################model is wrapped####################")
            # print(f"####################type(model):{type(model)}####################")
#########################################################
            if self.encoder_decoder_mode:
                if model.module.encoder.config.ctc_loss_reduction == "mean":
                    ctc_loss = ctc_loss.mean()
                elif model.module.encoder.config.ctc_loss_reduction == "sum":
                    # print(f"valid_num_labels={(inputs['labels'] >= 0).sum()}")
                    ctc_loss = ctc_loss.sum() / inputs["labels"].shape[0]
                    # cross_entropy loss默认就是batch mean
                    att_loss = att_loss.sum() / inputs["labels"].shape[0]
                    if not torch.isnan(att_loss):
                        # 正常情况下att_loss不为nan
                        loss = ctc_weight*ctc_loss + (1.0-ctc_weight)*att_loss
                        # loss = 1.0*ctc_loss + 0.0*att_loss
                    else:
                        # print("!"*20,"att_loss is nan , let loss = ctc_loss, att_loss=0","!"*20)
                        loss = ctc_loss + 0.0*att_loss
                        att_loss = torch.tensor(0.0)
                    # loss_tuple = (ctc_weight*ctc_loss,(1-ctc_weight)*att_loss)
                    # print(loss, ctc_loss, att_loss)
#                     print(f"joint ctc_att_loss={loss.item()}")
                else:
                    raise ValueError(f"{model.encoder.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
            else:
                if model.module.config.ctc_loss_reduction == "mean":
                    loss = loss.mean()
                elif model.module.config.ctc_loss_reduction == "sum":
                    # print(f"valid_num_labels={(inputs['labels'] >= 0).sum()}")
                    loss = loss.sum() / inputs["labels"].shape[0]
                     # loss = loss.sum() / (inputs['labels'] >= 0).sum()
                    # print("mean by batch",loss)
                else:
                    raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
        else :
            # print("####################model is not wrapped####################")
            # print(f"####################type(model):{type(model)}####################")        
            if self.encoder_decoder_mode:
                if model.encoder.config.ctc_loss_reduction == "mean":
                    ctc_loss = ctc_loss.mean()
                elif model.encoder.config.ctc_loss_reduction == "sum":
                    # print(f"valid_num_labels={(inputs['labels'] >= 0).sum()}")
                    ctc_loss = ctc_loss.sum() / inputs["labels"].shape[0]
#                     print(f"mean ctc_loss={loss}")
                    loss_tuple = (ctc_loss, att_loss)
                    loss = ctc_weight*ctc_loss + (1-ctc_weight)*att_loss
#                     print(f"joint ctc_att_loss={loss.item()}")
                else:
                    raise ValueError(f"{model.encoder.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
            else:
                if model.config.ctc_loss_reduction == "mean":
                    loss = loss.mean()
                elif model.config.ctc_loss_reduction == "sum":
                    # print(f"valid_num_labels={(inputs['labels'] >= 0).sum()}")
                    loss = loss.sum() / inputs["labels"].shape[0]
                else:
                    raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")        

       
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            # print(f"final loss=  {loss}")
            if self.encoder_decoder_mode:
                ctc_loss = ctc_weight*ctc_loss / self.args.gradient_accumulation_steps
                att_loss = (1-ctc_weight)*att_loss / self.args.gradient_accumulation_steps
                loss_tuple = (ctc_loss, att_loss)

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()
        if self.encoder_decoder_mode:
            return loss.detach(),loss_tuple
        
        return loss.detach()

def show_args(args):
    print('\n'.join(['%s:%s' % item for item in args.__dict__.items()]))
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    configure_logger(model_args, training_args)
    stars = "*" * 20
    logger.info(f"{stars}model_args:{stars}\n{model_args}")
    logger.info(f"{stars}data_args:{stars}\n{data_args}")
    logger.info(f"{stars}training_args:{stars}\n{training_args}")

    orthography = Orthography.from_name(data_args.orthography.lower())

    # processor = orthography.create_processor(model_args)
    # model = Wav2Vec2ForCTC.from_pretrained(
        # model_args.model_name_or_path,
        # cache_dir="/data2_from_58175/huggingface/models",
        # gradient_checkpointing=model_args.gradient_checkpointing,
        # vocab_size=32,
    # ) 
    # 10h labelled ft drop=0.1
    # 20h pseudo_labelled ft drop=0.2
    model = Wav2Vec2ForCTC.from_pretrained(model_args.model_name_or_path,layerdrop=0.1)
    pseudo_model = Wav2Vec2ForCTC.from_pretrained(model_args.model_name_or_path,layerdrop=0.1) if model_args.pseudo_onthefly else None
    if model_args.encoder_decoder_mode:
        # encoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2-large-lv60"
        # decoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/decoder_init"
    
        encoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/encoder"
        decoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/decoder"
        encoder = Wav2Vec2ForCTC.from_pretrained(encoder_model_path)
        decoder = AutoModelForCausalLM.from_pretrained(decoder_model_path)
        model = Wav2vec2_Gpt2(encoder=encoder,decoder=decoder)
        encoder_ = Wav2Vec2ForCTC.from_pretrained(encoder_model_path)
        decoder_ = AutoModelForCausalLM.from_pretrained(decoder_model_path)              
        state_dict = torch.load("/data2_from_58175/wav2vec2_output/fisher_joint/nfex-checkpoint-24000/pytorch_model.bin",map_location="cpu")
        model.load_state_dict(state_dict)   
        logger.info(f"{stars}use Wav2vec2_Gpt2 model{stars} ")

    if model_args.reinit_lm_head:
        nn.init.normal_(model.lm_head.weight)
        logger.info(f"{stars}reinitial lm_head{stars}")

    if not model_args.use_gpt_tokenizer:
        processor = Wav2Vec2Processor.from_pretrained("/data2_from_58175/huggingface/models/wav2vec2-large-960h-lv60-self") 
    else:    
        processor = Wav2Vec2Processor_Gpt_Tokenizer.from_pretrained(
                                                    pretrained_model_name_or_path="/data2_from_58175/huggingface/models/wav2vec2-large-960h-lv60-self",
                                                    gpt_tokenizer_name_or_path="/data2_from_58175/huggingface/models/tiny-gpt2-tokenizer",
                                                   )
        logger.info(f"{stars}use_gpt_tokenizer,vocab_size={processor.tokenizer.vocab_size}{stars}")
        model.lm_head=nn.Linear(1024,50257)

    wer_metric = datasets.load_metric("/data2_from_58175/huggingface/metrics/wer")


    prepare_train_path = "/home/data/fisher_swbd_nodup_onlyspeech/swbd_simu_100h"  
    # prepare_validation_path = "/data2_from_58175/huggingface/datasets/aishell/aishell_validation_prepare_4239_nospace"
    prepare_validation_path = "/data2_from_58175/huggingface/datasets/swbd_nodup/test_10h"
    prepare_test_path = "/home/data/fisher_swbd_nodup_onlyspeech/swbdtest5h"


    # speech和input_values取一个就好（默认需要input_values），目前硬盘里的dataset是二者都包含。做变速扩增，需要的是speech而不是input_values
    # 实际上再硬盘里可以不存input_values，只需要在data_collector中对speech做归一化即可，这样还能节省一大半空间
    # train_dataset = load_from_disk(prepare_train_path).remove_columns(["input_values"]) if data_args.speed_perturb else load_from_disk(prepare_train_path).remove_columns(["speech"])
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"{stars}loading train_dataset{stars}")
    train_dataset = load_from_disk(prepare_train_path)
    # 当speech和inputvalues都存在时，除去input_values
    # 只取fisher做训练

    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    if ("speech" in train_dataset.features) and ("input_values" in train_dataset.features):
        train_dataset = train_dataset.remove_columns(["input_values"])  
    else:
        train_dataset = train_dataset    

    val_dataset = load_from_disk(prepare_test_path,keep_in_memory=True).remove_columns(['sampling_rate', 'seg_end', 'seg_start', 'text'])
    
    # val_dataset = load_from_disk(prepare_validation_path)
    if ("speech" in val_dataset.features) and ("input_values" in val_dataset.features):
        val_dataset = val_dataset.remove_columns(["input_values"])  
    else:
        val_dataset = val_dataset       
    # val_dataset = load_from_disk(prepare_validation_path).select(np.arange(256))
    # 0-6850是1h
    train_dataset_teacher = train_dataset.select(range(6850)).remove_columns(['file', 'id'])
    # train_dataset_teacher = train_dataset.select(range(6850,6850+680)).remove_columns(['file', 'id'])
    logger.info(f"train_dataset_teacher:\n{train_dataset_teacher}\n")
    # train_dataset_sp11 = load_from_disk("/home/data/fisher_swbd_nodup_onlyspeech/swbd_simu_1h_sp11").remove_columns(['file', 'id'])
    # train_dataset_sp9 = load_from_disk("/home/data/fisher_swbd_nodup_onlyspeech/swbd_simu_1h_sp9").remove_columns(['file', 'id'])
    train_dataset = train_dataset_teacher
    # train_dataset = datasets.concatenate_datasets([train_dataset_sp11,train_dataset_sp9,train_dataset_teacher])
    
    logger.info(f"{stars}get dataset over{stars}")
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))   

    train_dataset_total_dur = sum(train_dataset["length"])/3600
    train_dataset_maxlength = sorted(train_dataset["length"],reverse=True)[0]
    
    
    val_total_dur = sum(val_dataset["length"])/3600
    val_maxlength = sorted(val_dataset["length"],reverse=True)[0]
    

    logger.info(f"train_dataset:\n{train_dataset}\n")
    logger.info(f"total duration of train_dataset:\n{train_dataset_total_dur} hours\n")
    logger.info(f"maxlength of train_dataset:\n{train_dataset_maxlength} s\n")
    

    logger.info(f"val_dataset:\n{val_dataset}\n")
    logger.info(f"total duration of val_dataset:\n{val_total_dur} hours\n")
    logger.info(f"maxlength of val_dataset:\n{val_maxlength} s\n")


    # 变速在data_collator中实现，默认使用torchaudio，但由于一些原因，需要把speech值作为输入，而input_values值可以忽略，input_column_change在_remove_unused_columns作用
    logger.info(f"{stars}do speech perpturbation{stars}") if data_args.speed_perturb else None

    data_collator = DataCollatorCTCWithPadding_Speed_Perturb(processor=processor, padding=True) if data_args.speed_perturb else DataCollatorCTCWithPadding(processor=processor, padding=True)
    # 验证集不进行在线增强
    data_collator_eval = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        # gpt_tokenizer ,skip_special_tokens=True
        pred_str = processor.batch_decode(pred_ids)
        
        # we do not want to group tokens when computing the metrics
        # RUNNING ->RUNING
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        if logger.isEnabledFor(logging.WARN):
            for reference, predicted in zip(label_str[50:60], pred_str[50:60]):
                logger.warn(f'reference: "{reference}"')
                logger.warn(f'predicted: "{predicted}"')
    #             if orthography.untransliterator is not None:
    #                 logger.debug(f'reference (untransliterated): "{orthography.untransliterator(reference)}"')
    #                 logger.debug(f'predicted (untransliterated): "{orthography.untransliterator(predicted)}"')

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    if model_args.freeze_feature_extractor:
        logger.info(f"{stars}freeze parameters of freeze_feature_extractor{stars} ")
        model.freeze_feature_extractor() if not model_args.encoder_decoder_mode else model.encoder.freeze_feature_extractor()
    if model_args.freeze_ALN:
        logger.info(f"{stars}freeze parameters of freeze_feature_extractor and feed_forward{stars} ")
        model.freeze_ALN() if not model_args.encoder_decoder_mode else model.encoder.freeze_ALN()
    if model_args.freeze_all_except_lm:
        logger.info(f"{stars}freeze all parameters of the model except lm_head{stars} ")
        model.freeze_all_except_lm() if not model_args.encoder_decoder_mode else model.encoder.freeze_all_except_lm()
    if model_args.freeze_all_except_feature_extractor:
        logger.info(f"{stars}freeze all parameters of the model except feature_extractor{stars} ")
        model.freeze_all_except_feature_extractor() if not model_args.encoder_decoder_mode else model.encoder.freeze_all_except_feature_extractor()
    if model_args.freeze_decoder:
        logger.info(f"{stars}freeze all parameters of the decoder when in encoder_decoder_mode{stars} ")
        if model_args.encoder_decoder_mode:
            model.freeze_decoder() 
        else:
            raise ValueError(
                f"freeze decoder is only allowed when encoder_decoder_mode is True"
            )
    if model_args.freeze_w2v2forctc:
        logger.info("!!!!!!!!!!!!freeze all parameters of the w2v2forctc when in encoder_decoder_mode!!!!!!!!!!!! ")
        if model_args.encoder_decoder_mode:
            model.encoder._freeze_parameters()
        else:
            raise ValueError(
                f"freeze w2v2forctc is only allowed when encoder_decoder_mode is True"
            )
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        data_collator_eval=data_collator_eval,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        train_dataset_teacher=train_dataset_teacher,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        processor=processor,
        pseudo_model=pseudo_model,
        teacher_model=None,
        encoder_decoder_mode=model_args.encoder_decoder_mode,
    ) 
    trainer.train(resume_from_checkpoint = training_args.resume_from_checkpoint)
if __name__ == "__main__":
    main()