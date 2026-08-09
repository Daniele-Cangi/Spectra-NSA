from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable, Mapping


def validate_disjoint_groups(split_groups: Mapping[str, Iterable[Hashable]]) -> None:
    """Reject source-item or generator leakage across dataset splits."""

    owners: dict[Hashable, list[str]] = defaultdict(list)
    for split_name, groups in split_groups.items():
        if not split_name.strip():
            raise ValueError("split names cannot be empty")
        for group in set(groups):
            owners[group].append(split_name)

    overlaps = {
        group: tuple(sorted(splits))
        for group, splits in owners.items()
        if len(splits) > 1
    }
    if overlaps:
        preview = ", ".join(
            f"{group!r}:{'/'.join(splits)}"
            for group, splits in list(
                sorted(overlaps.items(), key=lambda item: repr(item[0]))
            )[:5]
        )
        raise ValueError(f"group leakage detected across splits: {preview}")
