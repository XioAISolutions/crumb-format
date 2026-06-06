"""TurboVec-backed vector index for crumb context pulling.

This is the fast retrieval backend for :mod:`crumb_wavelm.context_pull`.
The wave-field model already turns each crumb section into a spectral
signature (``|rFFT(scatter(tokens))|``) — a fixed-length float vector.
Those signatures *are* embeddings, so we can hand them straight to
`turbovec <https://github.com/RyanCodrai/turbovec>`_, a local vector
index implementing Google Research's TurboQuant algorithm. No external
embedding service, no training phase — it keeps Crumb LLM's "runs
entirely locally" property while replacing the O(N) cosine scan in
``context_pull.score_sections`` with a compressed, sub-linear index.

turbovec is optional. It rides along with the ``[llm]`` extra; when it
is not installed, :func:`turbovec_available` returns ``False`` and the
caller falls back to the pure-Python cosine loop. Nothing here imports
torch — only numpy, which the ``[llm]`` extra already provides.

turbovec's ``search`` returns *inner products*, so we L2-normalize every
vector before indexing and before querying. Inner product on unit
vectors equals cosine similarity, which preserves the exact ranking
semantics the spectral matcher had before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:  # numpy ships with the [llm] extra; turbovec needs it too.
    import numpy as np
except Exception:  # pragma: no cover - numpy absent
    np = None  # type: ignore

try:
    from turbovec import IdMapIndex
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - turbovec absent
    IdMapIndex = None  # type: ignore
    _IMPORT_ERROR = exc

# Default quantization width. 4 bits is turbovec's headline setting:
# ~8x compression versus float32 with negligible recall loss on
# normalized vectors.
DEFAULT_BIT_WIDTH = 4


def turbovec_available() -> bool:
    """True when both turbovec and numpy can be imported."""
    return IdMapIndex is not None and np is not None


def _padded_dim(dim: int) -> int:
    """turbovec requires the index dim to be a positive multiple of 8."""
    return ((dim + 7) // 8) * 8


def _prepare(mat: "np.ndarray", padded_dim: int) -> "np.ndarray":
    """L2-normalize rows, then zero-pad to ``padded_dim``.

    Normalizing makes turbovec's inner-product score equal cosine
    similarity. Zero-padding the trailing dims is lossless: it changes
    neither the L2 norm nor any inner product, so cosine is preserved."""
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mat = mat / norms
    if mat.shape[1] < padded_dim:
        pad = np.zeros((mat.shape[0], padded_dim - mat.shape[1]), dtype=np.float32)
        mat = np.concatenate([mat, pad], axis=1)
    return np.ascontiguousarray(mat, dtype=np.float32)


class TurbovecVectorIndex:
    """Thin wrapper over ``turbovec.IdMapIndex`` keyed by section position.

    Sections are indexed under sequential integer ids (``0..N-1``) that
    map back to the caller's section list, so duplicate content hashes
    can never collide. Vectors are the spectral signatures, L2-normalized
    on the way in and on the way out.
    """

    def __init__(self, dim: int, bit_width: int = DEFAULT_BIT_WIDTH) -> None:
        if not turbovec_available():  # pragma: no cover - guarded by callers
            raise RuntimeError(
                "turbovec is not installed; run `pip install 'crumb-format[llm]'` "
                f"(import error: {_IMPORT_ERROR})"
            )
        self.dim = dim                      # true signature length
        self.padded_dim = _padded_dim(dim)  # turbovec index dim (multiple of 8)
        self.bit_width = bit_width
        self._index = IdMapIndex(dim=self.padded_dim, bit_width=bit_width)
        self._n = 0

    @classmethod
    def from_signatures(
        cls,
        signatures: list[list[float]],
        bit_width: int = DEFAULT_BIT_WIDTH,
    ) -> "TurbovecVectorIndex":
        """Build an index from a list of equal-length spectral signatures."""
        if not signatures:
            raise ValueError("cannot build a turbovec index from zero signatures")
        dim = len(signatures[0])
        self = cls(dim=dim, bit_width=bit_width)
        vecs = _prepare(np.asarray(signatures, dtype=np.float32), self.padded_dim)
        ids = np.arange(len(signatures), dtype=np.uint64)
        self._index.add_with_ids(vecs, ids)
        self._n = len(signatures)
        return self

    def search(self, query_signature: list[float], k: int) -> list[tuple[int, float]]:
        """Return ``[(section_index, cosine_score), ...]`` for the top-k matches."""
        if self._n == 0:
            return []
        k = min(k, self._n)
        q = _prepare(np.asarray(query_signature, dtype=np.float32), self.padded_dim)
        scores, ids = self._index.search(q, k)
        out: list[tuple[int, float]] = []
        for sid, score in zip(ids[0].tolist(), scores[0].tolist()):
            out.append((int(sid), float(score)))
        return out

    def __len__(self) -> int:
        return self._n
