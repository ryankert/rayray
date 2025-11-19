"""Array namespace for expression operations on fixed-size array-typed columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow
import pyarrow.compute as pc

from ray.data.datatype import DataType
from ray.data.expressions import pyarrow_udf

if TYPE_CHECKING:
    from ray.data.expressions import Expr, UDFExpr


@dataclass
class _ArrayNamespace:
    """Namespace for fixed-size array operations on expression columns.

    This namespace provides methods for operating on fixed-size array-typed columns
    using PyArrow compute functions.

    Example:
        >>> from ray.data.expressions import col
        >>> # Flatten fixed-size array to list
        >>> expr = col("embeddings").arr.flatten()
        >>> # Convert fixed-size array to list
        >>> expr = col("embeddings").arr.to_list()
    """

    _expr: Expr

    def flatten(self) -> "UDFExpr":
        """Flatten fixed-size array to a list.

        Returns:
            UDFExpr that converts fixed-size array to a list.
        """
        # Infer return type from the array's value type
        return_dtype = DataType(object)  # fallback
        if self._expr.data_type.is_arrow_type():
            arrow_type = self._expr.data_type.to_arrow_dtype()
            if pyarrow.types.is_fixed_size_list(arrow_type):
                value_type = arrow_type.value_type
                # Return a list of the same value type
                return_dtype = DataType.from_arrow(pyarrow.list_(value_type))

        @pyarrow_udf(return_dtype=return_dtype)
        def _arr_flatten(arr: pyarrow.Array) -> pyarrow.Array:
            # Convert fixed-size list to regular list
            if pyarrow.types.is_fixed_size_list(arr.type):
                # Use list_flatten to convert fixed-size list array to a flat array,
                # then reconstruct as list arrays
                offsets = []
                offset = 0
                for i in range(len(arr)):
                    offsets.append(offset)
                    if arr[i].is_valid:
                        offset += len(arr[i])
                offsets.append(offset)
                
                flat_values = pc.list_flatten(arr)
                return pyarrow.ListArray.from_arrays(offsets, flat_values)
            else:
                return arr

        return _arr_flatten(self._expr)

    def to_list(self) -> "UDFExpr":
        """Convert fixed-size array to a variable-length list.

        This is an alias for flatten().

        Returns:
            UDFExpr that converts fixed-size array to a list.
        """
        return self.flatten()
