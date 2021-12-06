from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel,GPT2Config,BertGenerationConfig
import torch
from typing import Optional, Tuple, List
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutput
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, Wav2Vec2ForCTC
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
            >>> sos_id = 1
            >>> eos_id = 2
            >>> ignore_id = -100
            >>> ys_pad
            tensor([[ 2,  3,  4,    5,    6],
                    [ -1, 7,  8, -100, -100],
                    [ 9, 10, 11, -100, -100]], dtype=torch.int32)
            >>> out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
            >>> ys_in
            tensor([[ 1, 2,  3,    4,    5,    6],
                    [ 1, 7,  8, -100, -100, -100],
                    [ 1, 9, 10, -100, -100, -100]], dtype=torch.int32)
            >>> ys_out
            tensor([[ 1, 2,  3, 4,    5,    6],
                    [ 1, 7,  8, 2, -100, -100],
                    [ 1, 9, 10, 2, -100, -100]], dtype=torch.int32)
        """
        # 实时伪标签以pseudo_label_id=-1开头，所以在add_sos_eos需要把-1删去
        pseudo_label_id = -1
        _sos = torch.tensor([sos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        _eos = torch.tensor([eos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        ys = [y[(y != ignore_id) * (y != pseudo_label_id)] for y in ys_pad]  # parse padded ys
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
        forward_only_encoder=None, 
        decoder_teacher_logits=None,
        encoder_teacher_logits=None,
        decoder_entropy_loss=False,
        encoder_entropy_loss=False,
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
                encoder_teacher_logits=encoder_teacher_logits,
                encoder_entropy_loss=encoder_entropy_loss,
                **kwargs_encoder,
            )

#         print(len(encoder_all_hidden_states))
        encoder_logits = encoder_outputs.logits
        encoder_hidden_states = encoder_outputs.hidden_states[-1]
#         encoder_hidden_states.retain_grad()
#         print(encoder_hidden_states.shape)
        self.encoder_hidden_states=encoder_hidden_states
#         print(self.encoder_hidden_states.requires_grad)

        
        # 传给encoder的labels不能直接传给decoder的input_ids及labels
        # 对于decoder_labels，对labels进行操作: labels中（非-100）首尾添加<bos>、<eos>
        # 对于decoder_input_ids，在decoder_labels基础上，需要用0替代-100，否则nn.embedding(-100)越界
        # 用0取代后pad的token不会对句子造成影响（attention_mask），一般输入token中没有0
        # 在decoder_labels基础上取dec_input_attention_mask
        # print(f"labels={labels}") if labels is not None else None
#         print(f"attention_mask.shape={attention_mask.shape}")        
        sos_id = self.decoder.config.bos_token_id
        eos_id = self.decoder.config.eos_token_id
#         print(f"sos_id={sos_id}")
#         print(f"eos_id={eos_id}")
        ignore_id = -100
        if not forward_only_encoder:
            pseudo_mask = labels==-1
            # 得到伪标签样本在当前batch的行索引值
            pseudo_rows = pseudo_mask.nonzero()[:,0]#.tolist()
            pseudo_rows = None if pseudo_rows==[] else pseudo_rows
            # print(f"pseudo_rows = {pseudo_rows}")
            # pseudo_rows [1,5,6,7,8]代表第1，5，6，7，8为伪标签样本，为空[]则代表不存在伪标签样本
            # 传给decoder，用于分开二者loss的计算，一部分使用hard，一部分使用soft
            dec_labels = self.add_sos_eos(labels, sos_id, eos_id, ignore_id)
            dec_input_attention_mask = dec_labels.ne(-100)
            dec_input_ids = dec_labels.masked_fill(~dec_input_attention_mask, 0)
            # print(f"dec_labels={dec_labels}")
#             print(f"dec_input_attention_mask={dec_input_attention_mask}")
            # print(f"dec_input_ids={dec_input_ids}")
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
#         self.encoder_attention_mask = enc_frame_attention_mask
#         self.encoder_hidden_states = encoder_hidden_states
        # Decode
        vocab_size = self.encoder.config.vocab_size
        if not forward_only_encoder:
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
                decoder_teacher_logits=decoder_teacher_logits,
                pseudo_rows = pseudo_rows,
                decoder_entropy_loss = decoder_entropy_loss,
                vocab_size = vocab_size,
                **kwargs_decoder,
            )
            if not return_dict:
                return decoder_outputs + encoder_outputs

            return Seq2SeqLMOutput(
                loss=(encoder_outputs.loss,decoder_outputs.loss),
                logits=(encoder_outputs.logits,decoder_outputs.logits),
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            ) 
        return encoder_hidden_states.detach(),enc_frame_attention_mask.detach(),encoder_logits.detach()





def main():
    encoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/encoder"
    decoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/decoder"
    encoder = Wav2Vec2ForCTC.from_pretrained(encoder_model_path)
    decoder = AutoModelForCausalLM.from_pretrained(decoder_model_path)
    model = Wav2vec2_Gpt2(encoder=encoder,decoder=decoder)
    print(model)
    
# 在被import时__name__不等于__main__，则不会进入main(), 当直接执行本脚本时，__name__=__main__
if __name__ == "__main__":
    main()