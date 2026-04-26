import random
from codebert_head_interpretability.schemas.analysis import HeadAnalysisResult
from codebert_head_interpretability.schemas.code_query import CodeQueryModel
from .code_query import CodeQueryPipeline


class MismatchPipeline(CodeQueryPipeline):
    def run(self, split="train", max_examples=100, output_dir="outputs"):
        ds = self.dataset.load(split)

        all_results: list[HeadAnalysisResult] = []
        examples = list(self.dataset.to_examples(ds, max_examples))
        examples = self._generate_mismatched_examples(examples)

        print("\nProcessing dataset...\n")

        for i, example in enumerate(
            self.dataset.to_examples(ds, max_examples=max_examples)
        ):
            try:
                results = self.process_example(example)
                all_results.extend(results)

            except Exception as e:
                print(f"Skipping example {i}: {e}")
                continue

            if i % 10 == 0 and i > 0:
                print(f"Processed {i} examples...")

        print("\nGenerating visualizations...\n")

        self._visualize(all_results, output_dir)

        print(f"\nAll outputs saved to '{output_dir}/'\n")

    def _generate_mismatched_examples(
        self,
        examples: list[CodeQueryModel],
    ) -> list[CodeQueryModel]:

        queries = [ex.query for ex in examples]
        codes = [ex.code for ex in examples]

        shuffled_queries = queries[:]

        while True:
            random.shuffle(shuffled_queries)

            if all(q1 != q2 for q1, q2 in zip(queries, shuffled_queries)):
                break

        return [
            CodeQueryModel(code=code, query=query)
            for code, query in zip(codes, shuffled_queries)
        ]
