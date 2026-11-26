import os


def resolve_fewshot_cache_dir(cfg, dataset_dir, dataset_name):
    """Return a dataset-scoped few-shot cache directory.

    An empty ``DATASET.FEWSHOT_CACHE_ROOT`` preserves the original behavior.
    When a root is configured, ``DATASET.NAME`` is used as a separate
    namespace so identically named ``shot_<n>-seed_<s>.pkl`` files from
    different datasets cannot overwrite one another.
    """
    cache_root = str(getattr(cfg.DATASET, "FEWSHOT_CACHE_ROOT", "")).strip()
    if not cache_root:
        return os.path.join(dataset_dir, "split_fewshot")

    configured_name = str(getattr(cfg.DATASET, "NAME", "")).strip()
    if configured_name != dataset_name:
        raise ValueError(
            "Few-shot cache namespace mismatch: "
            f"DATASET.NAME={configured_name!r}, expected {dataset_name!r}"
        )

    if dataset_name in {".", ".."} or "/" in dataset_name or "\\" in dataset_name:
        raise ValueError(f"Unsafe DATASET.NAME for cache namespace: {dataset_name!r}")

    cache_root = os.path.abspath(os.path.expanduser(cache_root))
    return os.path.join(cache_root, dataset_name, "split_fewshot")
