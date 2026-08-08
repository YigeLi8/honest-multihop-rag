Raw downloads go to raw/, normalized jsonl to processed/ (both gitignored).

    python -m data.download
    python -m data.prepare_hotpotqa
    python -m data.prepare_2wiki
    python -m data.prepare_beir

One record per question:

    {"id", "question", "answer", "hops",
     "chunks": [{"chunk_id": "Title::3", "title", "sent_idx", "text"}, ...],
     "gold_chunk_ids": [...],
     "gold_supporting_facts": [{"title", "sent_idx"}, ...]}

2Wiki records also carry "evidence_triples" and "reasoning_path". Gold chunk ids
come from a heuristic mapping of the datasets' supporting facts onto chunks
(src/eval/gold_mapping.py); the mapping error gets measured and reported, since
every precision number downstream inherits it.
