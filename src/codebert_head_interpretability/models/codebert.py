import torch
from transformers import RobertaTokenizer, RobertaModel

from codebert_head_interpretability.models.base import BaseModel
from codebert_head_interpretability.schemas.model_output import (
    ModelOutput,
    WindowOutput,
    ModelToken,
)
from codebert_head_interpretability.utils.sliding_window import (
    create_sliding_windows,
    build_query_code_window,
)


class CodeBertModel(BaseModel):
    def __init__(self, model_name: str = "microsoft/codebert-base"):

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        self.model = RobertaModel.from_pretrained(
            model_name,
            output_attentions=True,
        )

        self.max_length = 512
        self.stride = 256

    def _build_model_tokens(
        self,
        tokens: list[str],
        offsets: list[tuple[int, int]],
    ) -> list[ModelToken]:

        model_tokens: list[ModelToken] = []

        for i, (tok, (start, end)) in enumerate(zip(tokens, offsets)):
            model_tokens.append(
                ModelToken(
                    text=tok,
                    start=start,
                    end=end,
                    index=i,
                )
            )

        return model_tokens

    def run_code(self, code: str) -> ModelOutput:

        encoding = self.tokenizer(
            code,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        input_ids: list[int] = encoding["input_ids"]
        offsets: list[tuple[int, int]] = encoding["offset_mapping"]

        windows = create_sliding_windows(
            input_ids=input_ids,
            offsets=offsets,
            max_length=self.max_length - 2,  # reserve <s>, </s>
            stride=self.stride,
        )

        all_windows: list[WindowOutput] = []

        for window_ids, window_offsets in windows:
            input_ids_full = (
                [self.tokenizer.cls_token_id]
                + window_ids
                + [self.tokenizer.sep_token_id]
            )

            attention_mask = [1] * len(input_ids_full)

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids_full)

            # offsets: add dummy offsets for special tokens
            offsets_full = [(0, 0)] + window_offsets + [(0, 0)]

            model_tokens = self._build_model_tokens(tokens, offsets_full)

            input_tensor = torch.tensor([input_ids_full])
            mask_tensor = torch.tensor([attention_mask])

            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
            )

            all_windows.append(
                WindowOutput(
                    tokens=model_tokens,
                    attentions=outputs.attentions,
                )
            )

        return ModelOutput(windows=all_windows)

    def run_query_code(self, query: str, code: str) -> ModelOutput:

        query_enc = self.tokenizer(
            query,
            add_special_tokens=False,
        )

        code_enc = self.tokenizer(
            code,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        query_ids: list[int] = query_enc["input_ids"]
        code_ids: list[int] = code_enc["input_ids"]
        code_offsets: list[tuple[int, int]] = code_enc["offset_mapping"]

        windows = build_query_code_window(
            tokenizer=self.tokenizer,
            query_ids=query_ids,
            code_ids=code_ids,
            code_offsets=code_offsets,
            max_length=self.max_length,
            stride=self.stride,
        )

        all_windows: list[WindowOutput] = []

        for input_ids_full, attention_mask, code_window_offsets in windows:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids_full)

            offsets_full: list[tuple[int, int]] = []

            code_offset_idx = 0

            for tok in tokens:
                if tok in ["<s>", "</s>"]:
                    offsets_full.append((0, 0))
                elif code_offset_idx < len(code_window_offsets):
                    offsets_full.append(code_window_offsets[code_offset_idx])
                    code_offset_idx += 1
                else:
                    offsets_full.append((0, 0))

            model_tokens = self._build_model_tokens(tokens, offsets_full)

            input_tensor = torch.tensor([input_ids_full])
            mask_tensor = torch.tensor([attention_mask])

            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
            )

            all_windows.append(
                WindowOutput(
                    tokens=model_tokens,
                    attentions=outputs.attentions,
                )
            )

        return ModelOutput(windows=all_windows)
