"""Shared helpers for RAGAS evaluation in Session 10 Activity 1."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd


def as_context_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def load_testset(path: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    if limit is not None:
        df = df.head(limit)
    return df


def run_rag_over_testset(graph, dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        result = graph.invoke({"question": row["user_input"]})
        rows.append(
            {
                "user_input": row["user_input"],
                "reference": row["reference"],
                "reference_contexts": as_context_list(row["reference_contexts"]),
                "retrieved_contexts": [
                    document.page_content for document in result["context"]
                ],
                "response": result["answer"],
            }
        )
    return rows


def run_ragas_sync(func, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(func, *args, **kwargs).result()


def run_ragas_async(async_function, *args, **kwargs):
    if asyncio.iscoroutine(async_function):
        return run_ragas_sync(asyncio.run, async_function)

    def invoke():
        return asyncio.run(async_function(*args, **kwargs))

    return run_ragas_sync(invoke)
