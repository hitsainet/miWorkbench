"""Discovery tools (category: read) — extractions and trainings."""

from typing import Annotated, Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import MiWorkbenchClient
from ..config import MCPSettings


def register(mcp: MCPServer, client: MiWorkbenchClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def list_extractions(
        status_filter: Annotated[Optional[str], Field(description="Comma-separated statuses, e.g. 'completed,failed'")] = None, limit: Annotated[int, Field(description="Max rows to return")] = 50, offset: Annotated[int, Field(description="Rows to skip, for paging")] = 0
    ) -> Any:
        """List feature-extraction jobs (the entry point for feature analysis).

        Each extraction harvested features from one SAE. Use its id with
        search_features, feature-group tools, and by-token queries.
        status_filter: comma-separated statuses (e.g. 'completed'). limit ≤ 100.
        """
        return await client.get(
            "/extractions", status_filter=status_filter, limit=min(limit, 100), offset=offset
        )

    @mcp.tool()
    async def get_extraction_summary(extraction_id: Annotated[str, Field(description="Extraction job id (extr_xxxxxxxx) from list_extractions — features belong to an EXTRACTION/SAE, not a training")]) -> Any:
        """Get one extraction job's detail: SAE, status, feature counts, config."""
        return await client.get(f"/extractions/{extraction_id}")

    @mcp.tool()
    async def list_trainings(limit: Annotated[int, Field(description="Max rows to return")] = 50, offset: Annotated[int, Field(description="Rows to skip, for paging")] = 0) -> Any:
        """List SAE training runs (id, model, status, hyperparameters, metrics).

        `/trainings` pages by page number, not offset (review R2D-2: the offset this tool
        used to send was ignored, so every page was page 1). The offset is rounded down to
        the start of its page and the rows before it on that page are dropped, so an offset
        that is not a multiple of limit returns fewer than limit rows.
        """
        limit = max(1, min(limit, 100))
        page_index, rows_before_offset = divmod(max(0, offset), limit)
        result = await client.get("/trainings", limit=limit, page=page_index + 1)
        if rows_before_offset and isinstance(result, dict) and isinstance(result.get("data"), list):
            result = {**result, "data": result["data"][rows_before_offset:]}
        return result

    # ── 2026-10-19 (MCP gap closure): the ids every other tool asks for ───────────────────
    #
    # `register_probe_dataset` takes a dataset id and steering/circuit tools take an SAE id, and
    # no tool could list either: an agent had to be handed them.

    @mcp.tool()
    async def get_training(
        training_id: Annotated[str, Field(description="SAE training id (train_xxxxxxxx) from list_trainings")],
    ) -> Any:
        """One SAE training run: model, dataset, hyperparameters, status, progress and final
        metrics."""
        return await client.get(f"/trainings/{training_id}")

    @mcp.tool()
    async def list_datasets(
        search: Annotated[Optional[str], Field(description="Match name or HuggingFace repo id")] = None,
        status: Annotated[Optional[str], Field(description="downloading | processing | ready | error")] = None,
        page: Annotated[int, Field(description="Page number, from 1", ge=1)] = 1,
        limit: Annotated[int, Field(description="Rows per page (1-100)", ge=1, le=100)] = 50,
    ) -> Any:
        """List miWorkbench's downloaded datasets. Their `id` (a UUID) is what
        `register_probe_dataset` takes as `dataset_id`. A dataset not here yet comes in with
        `download_dataset`."""
        return await client.get("/datasets", search=search, status=status, page=page,
                                limit=limit)

    @mcp.tool()
    async def get_dataset(
        dataset_id: Annotated[str, Field(description="Dataset id (UUID) from list_datasets")],
    ) -> Any:
        """One dataset: source, revision, splits, size and status. Poll this after
        `download_dataset`: the row's `status` says whether it finished. Check the columns and
        splits here before registering a probe dataset view over it."""
        return await client.get(f"/datasets/{dataset_id}")

    @mcp.tool()
    async def get_dataset_rows(
        dataset_id: Annotated[str, Field(description="Dataset id (UUID) from list_datasets — a download, not a probe dataset view")],
        split: Annotated[Optional[str], Field(description="The split to read; required when the download has several (get_dataset lists them)")] = None,
        offset: Annotated[int, Field(description="First row to return, by position in the split", ge=0)] = 0,
        limit: Annotated[int, Field(description="Rows per page (1-200)", ge=1, le=200)] = 100,
        columns: Annotated[Optional[List[str]], Field(description="Columns to return; omit for every column")] = None,
    ) -> Any:
        """A DOWNLOADED dataset's rows by position, paged — no probe dataset view needed. Read-only.

        Use it to read a new set before registering it, e.g. to prove it shares no row with earlier sets.
        Each row is `{position, values, truncated}`: a cell longer than 20,000 characters is cut and its
        column named in `truncated`. Page with `next_offset` until it is null. A download with several
        splits needs `split`; an unknown split or column is refused 422 with what exists. 409
        `rows_not_on_disk` means the download's files are not on disk (still downloading, or removed).
        For a registered view's rows with their mapped labels, use `get_probe_dataset_rows`."""
        params: Dict[str, Any] = {"offset": offset, "limit": limit}
        if split is not None:
            params["split"] = split
        if columns is not None:
            params["columns"] = ",".join(columns)
        return await client.get(f"/datasets/{dataset_id}/rows", **params)

    @mcp.tool()
    async def list_saes(
        search: Annotated[Optional[str], Field(description="Match name, description or HuggingFace repo")] = None,
        source: Annotated[Optional[str], Field(description="huggingface | local | trained")] = None,
        model_name: Annotated[Optional[str], Field(description="Only SAEs for this model name")] = None,
        limit: Annotated[int, Field(description="Rows to return (1-100)", ge=1, le=100)] = 50,
        skip: Annotated[int, Field(description="Rows to skip, for paging", ge=0)] = 0,
    ) -> Any:
        """List miWorkbench's SAEs (sae_xxxxxxxx) — the id steering, circuit and feature-index
        tools take. NOTE: miServer's SAE ids are a DIFFERENT namespace."""
        return await client.get("/saes", search=search, source=source, model_name=model_name,
                                limit=limit, skip=skip)

    @mcp.tool()
    async def get_sae(
        sae_id: Annotated[str, Field(description="miWorkbench SAE id (sae_xxxxxxxx)")],
    ) -> Any:
        """One SAE: model, layer, hook, width, architecture, normalisation and where it came
        from (trained, downloaded, or published to HuggingFace)."""
        return await client.get(f"/saes/{sae_id}")
