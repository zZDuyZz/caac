"""CAAC-Pool v1: one flat, source-independent input format for the CAAC tool.

A pool is a directory (or a pair of files) holding

  * ``rows.jsonl`` / ``rows.jsonl.gz`` / ``rows.csv`` -- one row per generation
    (question x model x draw), see :mod:`caac.pool_v1.schema`;
  * ``meta.json`` -- where the rows came from and how they were graded.

Nothing in the rows is specific to a model family or a benchmark. Which analyses
the data allows is not assumed but detected (:func:`caac.pool_v1.validate.levels`):

  L0  ``correct``                      -> SERVE / ESCALATE / ABSTAIN, risk only
  L1  + ``out_tokens`` or ``cost_usd`` -> Audit (GO / NO-GO / INCONCLUSIVE), early stop
  L2  + >=2 draws per question with ``answer_key`` -> SAMPLE MORE (voting)
  L3  + ``finish`` (stop / length)    -> truncation checks
  L4  + ``gpu_s`` or ``cost_usd``     -> cost in GPU time / money next to tokens

The research pools (:class:`caac.pool.schema.PoolRecord`) are converted by an
adapter; the core replay and certification code is unchanged.
"""

from caac.pool_v1.schema import FORMAT, VERSION, Row, Meta  # noqa: F401
from caac.pool_v1.io import read_rows, read_meta, write_rows, load_pool_v1  # noqa: F401
from caac.pool_v1.validate import validate, levels, Issue, Report  # noqa: F401
