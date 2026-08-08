"""RAGAS faithfulness / context precision against a local OpenAI-compatible
endpoint. LLM-as-judge, so it gets reported next to the gold metrics, not
instead of them."""


def evaluate(results, endpoint, judge_model):
    raise NotImplementedError
