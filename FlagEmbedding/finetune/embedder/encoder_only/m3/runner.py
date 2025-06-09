import os
import torch
import logging
from typing import Tuple
from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)
from huggingface_hub import snapshot_download

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderRunner, AbsEmbedderModel,
    AbsEmbedderDataArguments, EmbedderTrainerCallbackForDataRefresh
)
from .modeling import EncoderOnlyEmbedderM3Model
from .trainer import EncoderOnlyEmbedderM3Trainer
from .arguments import EncoderOnlyEmbedderM3ModelArguments, EncoderOnlyEmbedderM3TrainingArguments
from .load_model import get_model, save_merged_model
from pathlib import Path

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderM3Runner(AbsEmbedderRunner):
    """
    M3 model runner for finetuning.
    
    Args:
        model_args (EncoderOnlyEmbedderM3ModelArguments): Model arguments
        data_args (AbsEmbedderDataArguments): Data arguments.
        training_args (EncoderOnlyEmbedderM3TrainingArguments): Training arguments.
    """
    def __init__(
        self,
        model_args: EncoderOnlyEmbedderM3ModelArguments,
        data_args: AbsEmbedderDataArguments,
        training_args: EncoderOnlyEmbedderM3TrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
        self.model_args: EncoderOnlyEmbedderM3ModelArguments
        self.data_args: AbsEmbedderDataArguments
        self.training_args: EncoderOnlyEmbedderM3TrainingArguments

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        """Load the tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
            padding_size=self.model_args.padding_side,
            add_eos_token=self.model_args.add_eos_token
        )

        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info('Config: %s', config)

        resize = False
        if self.model_args.additional_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': self.model_args.additional_special_tokens}
            add_num = tokenizer.add_special_tokens(special_tokens_dict)
            if add_num > 0:
                resize = True
                logger.info(f"Add {add_num} special tokens to the tokenizer. Special tokens: {self.model_args.additional_special_tokens}")
            else:
                logger.warning(f"Special tokens {self.model_args.additional_special_tokens} already exists in the tokenizer.")

        model = EncoderOnlyEmbedderM3Model(
            get_model(self.model_args, self.training_args.output_dir, resize, len(tokenizer)),
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings,
            unified_finetuning=self.training_args.unified_finetuning,
            use_self_distill=self.training_args.use_self_distill,
            self_distill_start_step=self.training_args.self_distill_start_step
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        
        if self.training_args.fix_encoder:
            for k, v in model.named_parameters():
                if "colbert_linear" in k or 'sparse_linear' in k:
                    logging.info(f"train the parameters for {k}")
                else:
                    v.requires_grad = False
        
        return tokenizer, model

    def load_trainer(self) -> EncoderOnlyEmbedderM3Trainer:
        """Load the M3 trainer.

        Returns:
            EncoderOnlyEmbedderM3Trainer: M3 Trainer instance.
        """
        trainer = EncoderOnlyEmbedderM3Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        return trainer

    def run(self):
        """
        Run the finetune.
        """
        if not self.model_args.only_merge_lora_model:
            Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

            # Training
            self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
            self.trainer.save_model()

        # save merged model
        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)