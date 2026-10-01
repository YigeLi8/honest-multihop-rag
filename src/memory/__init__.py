"""Experience memory for retrieval (docs/plan.md, Part B).

Stage 1 only logs: one record per (question, hop) with the strategy used, what
came back, model-free features of the situation and the outcome against the
dataset's evidence labels. Nothing in here changes what the pipeline retrieves
or answers.
"""
