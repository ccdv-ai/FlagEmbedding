from dataclasses import dataclass, field
from typing import List, Optional

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderTrainingArguments,
    AbsEmbedderModelArguments
)

def default_target_modules() -> List[int]:
    return ['v_proj', 'q_proj', 'k_proj', 'gate_proj', 'down_proj', 'o_proj', 'up_proj']

@dataclass
class EncoderOnlyEmbedderM3ModelArguments(AbsEmbedderModelArguments):
    """
    Model argument class for M3.
    """
    colbert_dim: int = field(default=-1, metadata={"help": "Dim of colbert linear"})
    use_flash_attn: bool = field(default=False, metadata={"help": "If passed, will use flash attention to train the model."})
    use_lora: bool = field(
        default=True,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."}
    )
    target_modules: List[str] = field(
        default_factory=default_target_modules,
        metadata={"help": "The target modules to apply LORA."}
    )
    from_peft: str = field(
        default=None
    )
    modules_to_save: str = field(
        default=None
    )
    raw_peft: str = field(
        default=None
    )
    additional_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "additional special tokens", "nargs": "+"}
    )
    save_merged_lora_model: bool = field(
        default=False,
        metadata={"help": "If passed, will merge the lora modules and save the entire model."}
    )
    only_merge_lora_model: bool = field(
        default=False,
        metadata={"help": "If passed, will only merge the lora modules and save the entire model."}
    )


@dataclass
class EncoderOnlyEmbedderM3TrainingArguments(AbsEmbedderTrainingArguments):
    """
    Training argument class for M3.
    """
    unified_finetuning: bool = field(default=False, metadata={"help": "use unify fine-tuning"})
    use_self_distill: bool = field(default=False, metadata={"help": "use self-distill when using unify fine-tuning"})
    fix_encoder: bool = field(default=False, metadata={"help": "Freeze the parameters of encoder"})
    self_distill_start_step: int = field(default=-1, metadata={"help": "Num of step when using self-distill"})
