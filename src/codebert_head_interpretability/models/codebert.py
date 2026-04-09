from transformers import RobertaTokenizer, RobertaModel
from codebert_head_interpretability.models.base import BaseModel
from codebert_head_interpretability.schemas.model_output import (
    ModelOutput,
    ModelToken,
)


class CodeBertModel(BaseModel):
    def __init__(self, model_name: str = "microsoft/codebert-base"):

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        self.model = RobertaModel.from_pretrained(model_name, output_attentions=True)

    def _build_tokens(self, tokens, offsets):

        model_tokens = []

        for i, (tok, (start, end)) in enumerate(zip(tokens, offsets)):
            model_tokens.append(ModelToken(text=tok, start=start, end=end, index=i))

        return model_tokens

    def run_code(self, code: str) -> ModelOutput:

        encoding = self.tokenizer(
            code, return_offsets_mapping=True, return_tensors="pt"
        )

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        offsets = encoding["offset_mapping"][0].tolist()

        model_tokens = self._build_tokens(tokens, offsets)

        outputs = self.model(**encoding)

        attentions = outputs.attentions

        return ModelOutput(tokens=model_tokens, attentions=attentions)

    def run_query_code(self, query: str, code: str) -> ModelOutput:

        encoding = self.tokenizer(
            query,
            code,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
        )

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        offsets = encoding["offset_mapping"][0].tolist()

        model_tokens = self._build_tokens(tokens, offsets)

        outputs = self.model(**encoding)

        attentions = outputs.attentions

        return ModelOutput(tokens=model_tokens, attentions=attentions)
