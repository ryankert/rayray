"""Datetime namespace for expression operations on datetime-typed columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import pyarrow
import pyarrow.compute as pc

from ray.data.datatype import DataType
from ray.data.expressions import pyarrow_udf

if TYPE_CHECKING:
    from ray.data.expressions import Expr, UDFExpr


def _create_dt_udf(
    pc_func: Callable[..., pyarrow.Array], return_dtype: DataType
) -> Callable[..., "UDFExpr"]:
    """Helper to create a datetime UDF that wraps a PyArrow compute function.

    Args:
        pc_func: PyArrow compute function that takes (array, *positional, **kwargs)
        return_dtype: The return data type

    Returns:
        A callable that creates UDFExpr instances
    """

    def wrapper(expr: Expr, *positional: Any, **kwargs: Any) -> "UDFExpr":
        @pyarrow_udf(return_dtype=return_dtype)
        def udf(arr: pyarrow.Array) -> pyarrow.Array:
            return pc_func(arr, *positional, **kwargs)

        return udf(expr)

    return wrapper


@dataclass
class _DatetimeNamespace:
    """Namespace for datetime operations on expression columns.

    This namespace provides methods for operating on datetime-typed columns using
    PyArrow compute functions.

    Example:
        >>> from ray.data.expressions import col
        >>> # Extract year from timestamp
        >>> expr = col("timestamp").dt.year()
        >>> # Format timestamp as string
        >>> expr = col("timestamp").dt.strftime("%Y-%m-%d")
        >>> # Extract day of week
        >>> expr = col("timestamp").dt.day()
    """

    _expr: Expr

    # Extraction methods
    def year(self) -> "UDFExpr":
        """Extract the year from datetime values."""
        return _create_dt_udf(pc.year, DataType.int64())(self._expr)

    def month(self) -> "UDFExpr":
        """Extract the month from datetime values."""
        return _create_dt_udf(pc.month, DataType.int64())(self._expr)

    def day(self) -> "UDFExpr":
        """Extract the day from datetime values."""
        return _create_dt_udf(pc.day, DataType.int64())(self._expr)

    def hour(self) -> "UDFExpr":
        """Extract the hour from datetime values."""
        return _create_dt_udf(pc.hour, DataType.int64())(self._expr)

    def minute(self) -> "UDFExpr":
        """Extract the minute from datetime values."""
        return _create_dt_udf(pc.minute, DataType.int64())(self._expr)

    def second(self) -> "UDFExpr":
        """Extract the second from datetime values."""
        return _create_dt_udf(pc.second, DataType.int64())(self._expr)

    def millisecond(self) -> "UDFExpr":
        """Extract the millisecond from datetime values."""
        return _create_dt_udf(pc.millisecond, DataType.int64())(self._expr)

    def microsecond(self) -> "UDFExpr":
        """Extract the microsecond from datetime values."""
        return _create_dt_udf(pc.microsecond, DataType.int64())(self._expr)

    def nanosecond(self) -> "UDFExpr":
        """Extract the nanosecond from datetime values."""
        return _create_dt_udf(pc.nanosecond, DataType.int64())(self._expr)

    def day_of_week(self) -> "UDFExpr":
        """Extract the day of week from datetime values (0=Monday, 6=Sunday)."""
        return _create_dt_udf(pc.day_of_week, DataType.int64())(self._expr)

    def day_of_year(self) -> "UDFExpr":
        """Extract the day of year from datetime values."""
        return _create_dt_udf(pc.day_of_year, DataType.int64())(self._expr)

    def iso_year(self) -> "UDFExpr":
        """Extract the ISO year from datetime values."""
        return _create_dt_udf(pc.iso_year, DataType.int64())(self._expr)

    def iso_week(self) -> "UDFExpr":
        """Extract the ISO week from datetime values."""
        return _create_dt_udf(pc.iso_week, DataType.int64())(self._expr)

    def iso_calendar(self) -> "UDFExpr":
        """Extract ISO calendar (year, week, day) from datetime values."""
        # Returns a struct with iso_year, iso_week, iso_day_of_week
        return _create_dt_udf(pc.iso_calendar, DataType(object))(self._expr)

    def quarter(self) -> "UDFExpr":
        """Extract the quarter from datetime values."""
        return _create_dt_udf(pc.quarter, DataType.int64())(self._expr)

    # Formatting methods
    def strftime(self, format: str, locale: str = "C") -> "UDFExpr":
        """Format datetime values as strings.

        Args:
            format: strftime format string
            locale: Locale to use for formatting (default: "C")

        Returns:
            UDFExpr that formats datetime values as strings.
        """
        return _create_dt_udf(pc.strftime, DataType.string())(
            self._expr, format=format, locale=locale
        )

    def strptime(self, format: str, unit: str = "us") -> "UDFExpr":
        """Parse string values as datetime.

        Args:
            format: strptime format string
            unit: Time unit for the result (default: "us")

        Returns:
            UDFExpr that parses strings as datetime values.
        """
        return _create_dt_udf(pc.strptime, DataType.timestamp(unit))(
            self._expr, format=format, unit=unit
        )

    # Timezone methods
    def assume_timezone(self, timezone: str) -> "UDFExpr":
        """Assume a timezone for naive datetime values.

        Args:
            timezone: Timezone name (e.g., "UTC", "America/New_York")

        Returns:
            UDFExpr that adds timezone information.
        """
        return _create_dt_udf(pc.assume_timezone, DataType.timestamp("us", timezone))(
            self._expr, timezone=timezone
        )

    # Rounding methods
    def ceil(self, unit: str = "day") -> "UDFExpr":
        """Round datetime values up to the nearest unit.

        Args:
            unit: Time unit to round to (e.g., "day", "hour", "minute")

        Returns:
            UDFExpr that rounds datetime values up.
        """
        return _create_dt_udf(pc.ceil_temporal, DataType.timestamp("us"))(
            self._expr, unit=unit
        )

    def floor(self, unit: str = "day") -> "UDFExpr":
        """Round datetime values down to the nearest unit.

        Args:
            unit: Time unit to round to (e.g., "day", "hour", "minute")

        Returns:
            UDFExpr that rounds datetime values down.
        """
        return _create_dt_udf(pc.floor_temporal, DataType.timestamp("us"))(
            self._expr, unit=unit
        )

    def round(self, unit: str = "day") -> "UDFExpr":
        """Round datetime values to the nearest unit.

        Args:
            unit: Time unit to round to (e.g., "day", "hour", "minute")

        Returns:
            UDFExpr that rounds datetime values.
        """
        return _create_dt_udf(pc.round_temporal, DataType.timestamp("us"))(
            self._expr, unit=unit
        )
