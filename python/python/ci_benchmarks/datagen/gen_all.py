# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import logging

from lance.log import LOGGER

from ci_benchmarks.datagen.basic import gen_basic
from ci_benchmarks.datagen.count_rows import gen_count_rows
from ci_benchmarks.datagen.lineitems import gen_tcph
from ci_benchmarks.datagen.merge_insert import gen_merge_insert
from ci_benchmarks.datagen.wikipedia import gen_wikipedia


def setup_logging():
    """Set up logging to display to console with timestamps."""
    # Check if handler already exists (avoid duplicate handlers)
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)


if __name__ == "__main__":
    setup_logging()
    LOGGER.info("=" * 80)
    LOGGER.info("Starting dataset generation for all benchmarks")
    LOGGER.info("=" * 80)

    LOGGER.info("Generating basic dataset...")
    gen_basic()

    LOGGER.info("Generating TPC-H lineitem dataset...")
    gen_tcph()

    LOGGER.info("Generating Wikipedia dataset...")
    gen_wikipedia()

    LOGGER.info("Generating count_rows benchmark dataset...")
    gen_count_rows()

    LOGGER.info("Generating merge_insert benchmark datasets...")
    gen_merge_insert()

    LOGGER.info("=" * 80)
    LOGGER.info("All datasets generated successfully!")
    LOGGER.info("=" * 80)
