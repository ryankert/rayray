"""Map namespace for expression operations on map-typed columns."""

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
class _MapNamespace:
    """Namespace for map operations on expression columns.

    This namespace provides methods for operating on map-typed columns using
    PyArrow compute functions.

    Example:
        >>> from ray.data.expressions import col
        >>> # Get keys from map column
        >>> expr = col("user_data").map.keys()
        >>> # Get values from map column
        >>> expr = col("user_data").map.values()
    """

    _expr: Expr

    def keys(self) -> "UDFExpr":
        """Extract all keys from map values.

        Returns:
            UDFExpr that extracts keys as a list.
        """
        # Infer return type from the map's key type
        return_dtype = DataType(object)  # fallback
        if self._expr.data_type.is_arrow_type():
            arrow_type = self._expr.data_type.to_arrow_dtype()
            if pyarrow.types.is_map(arrow_type):
                key_type = arrow_type.key_type
                # Return a list of keys
                return_dtype = DataType.from_arrow(pyarrow.list_(key_type))

        @pyarrow_udf(return_dtype=return_dtype)
        def _map_keys(arr: pyarrow.Array) -> pyarrow.Array:
            return pc.map_keys(arr)

        return _map_keys(self._expr)

    def values(self) -> "UDFExpr":
        """Extract all values from map values.

        Returns:
            UDFExpr that extracts values as a list.
        """
        # Infer return type from the map's value type
        return_dtype = DataType(object)  # fallback
        if self._expr.data_type.is_arrow_type():
            arrow_type = self._expr.data_type.to_arrow_dtype()
            if pyarrow.types.is_map(arrow_type):
                item_type = arrow_type.item_type
                # Return a list of values
                return_dtype = DataType.from_arrow(pyarrow.list_(item_type))

        @pyarrow_udf(return_dtype=return_dtype)
        def _map_values(arr: pyarrow.Array) -> pyarrow.Array:
            return pc.map_values(arr)

        return _map_values(self._expr)
