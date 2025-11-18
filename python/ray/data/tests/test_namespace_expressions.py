"""Tests for list, string, and struct namespace expressions.

This module tests the namespace accessor methods (list, str, struct) that provide
convenient access to PyArrow compute functions through the expression API.
"""

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

import ray
from ray.data.expressions import col


def assert_df_equal(result: pd.DataFrame, expected: pd.DataFrame):
    """Assert dataframes are equal, ignoring dtype differences."""
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def _create_dataset(
    items_data: Any, dataset_format: str, arrow_table: pa.Table | None = None
):
    if dataset_format == "arrow":
        if arrow_table is not None:
            # Use pre-constructed arrow table (for complex types like structs)
            ds = ray.data.from_arrow(arrow_table)
        else:
            # Convert items to arrow table (infers types automatically)
            table = pa.Table.from_pylist(items_data)
            ds = ray.data.from_arrow(table)
    elif dataset_format == "pandas":
        if arrow_table is not None:
            # Convert arrow table to pandas
            df = arrow_table.to_pandas()
        else:
            # Create pandas DataFrame from items
            df = pd.DataFrame(items_data)
        ds = ray.data.from_blocks([df])
    return ds


# Pytest parameterization for all dataset creation formats
DATASET_FORMATS = ["pandas", "arrow"]


# ──────────────────────────────────────
# List Namespace Tests
# ──────────────────────────────────────


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestListNamespace:
    """Tests for list namespace operations."""

    def test_list_len(self, dataset_format):
        """Test list.len() returns length of each list."""

        data = [
            {"items": [1, 2, 3]},
            {"items": [4, 5]},
            {"items": []},
        ]
        ds = _create_dataset(data, dataset_format)
        result = ds.with_column("len", col("items").list.len()).to_pandas()
        expected = pd.DataFrame(
            {
                "items": [[1, 2, 3], [4, 5], []],
                "len": [3, 2, 0],
            }
        )
        assert_df_equal(result, expected)

    def test_list_get(self, dataset_format):
        """Test list.get() extracts element at index."""

        data = [
            {"items": [10, 20, 30]},
            {"items": [40, 50, 60]},
        ]
        ds = _create_dataset(data, dataset_format)
        result = ds.with_column("first", col("items").list.get(0)).to_pandas()
        expected = pd.DataFrame(
            {
                "items": [[10, 20, 30], [40, 50, 60]],
                "first": [10, 40],
            }
        )
        assert_df_equal(result, expected)

    def test_list_bracket_index(self, dataset_format):
        """Test list[i] bracket notation for element access."""

        data = [{"items": [10, 20, 30]}]
        ds = _create_dataset(data, dataset_format)
        result = ds.with_column("elem", col("items").list[1]).to_pandas()
        expected = pd.DataFrame(
            {
                "items": [[10, 20, 30]],
                "elem": [20],
            }
        )
        assert_df_equal(result, expected)


# ──────────────────────────────────────
# String Namespace Tests
# ──────────────────────────────────────


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
@pytest.mark.parametrize(
    "method_name,input_values,expected_results",
    [
        ("len", ["Alice", "Bob"], [5, 3]),
        ("byte_len", ["ABC"], [3]),
    ],
)
class TestStringLength:
    """Tests for string length operations."""

    def test_string_length(
        self, dataset_format, method_name, input_values, expected_results
    ):
        """Test string length methods."""

        data = [{"name": v} for v in input_values]
        ds = _create_dataset(data, dataset_format)

        method = getattr(col("name").str, method_name)
        result = ds.with_column("result", method()).to_pandas()

        expected = pd.DataFrame({"name": input_values, "result": expected_results})
        assert_df_equal(result, expected)


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
@pytest.mark.parametrize(
    "method_name,input_values,expected_values",
    [
        ("upper", ["alice", "bob"], ["ALICE", "BOB"]),
        ("lower", ["ALICE", "BOB"], ["alice", "bob"]),
        ("capitalize", ["alice", "bob"], ["Alice", "Bob"]),
        ("title", ["alice smith", "bob jones"], ["Alice Smith", "Bob Jones"]),
        ("swapcase", ["AlIcE"], ["aLiCe"]),
    ],
)
class TestStringCase:
    """Tests for string case conversion."""

    def test_string_case(
        self, dataset_format, method_name, input_values, expected_values
    ):
        """Test string case conversion methods."""

        data = [{"name": v} for v in input_values]
        ds = _create_dataset(data, dataset_format)

        method = getattr(col("name").str, method_name)
        result = ds.with_column("result", method()).to_pandas()

        expected = pd.DataFrame({"name": input_values, "result": expected_values})
        assert_df_equal(result, expected)


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
@pytest.mark.parametrize(
    "method_name,input_values,expected_results",
    [
        ("is_alpha", ["abc", "abc123", "123"], [True, False, False]),
        ("is_alnum", ["abc123", "abc-123"], [True, False]),
        ("is_digit", ["123", "12a"], [True, False]),
        ("is_space", ["   ", " a "], [True, False]),
        ("is_lower", ["abc", "Abc"], [True, False]),
        ("is_upper", ["ABC", "Abc"], [True, False]),
        ("is_ascii", ["hello", "hello😊"], [True, False]),
    ],
)
class TestStringPredicates:
    """Tests for string predicate methods (is_*)."""

    def test_string_predicate(
        self, dataset_format, method_name, input_values, expected_results
    ):
        """Test string predicate methods."""

        data = [{"val": v} for v in input_values]
        ds = _create_dataset(data, dataset_format)

        # Get the method dynamically
        method = getattr(col("val").str, method_name)
        result = ds.with_column("result", method()).to_pandas()

        expected = pd.DataFrame({"val": input_values, "result": expected_results})
        assert_df_equal(result, expected)


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
@pytest.mark.parametrize(
    "method_name,method_args,input_values,expected_values",
    [
        ("strip", (), ["  hello  ", " world "], ["hello", "world"]),
        ("strip", ("x",), ["xxxhelloxxx"], ["hello"]),
        ("lstrip", (), ["  hello  "], ["hello  "]),
        ("rstrip", (), ["  hello  "], ["  hello"]),
    ],
)
class TestStringTrimming:
    """Tests for string trimming operations."""

    def test_string_trimming(
        self, dataset_format, method_name, method_args, input_values, expected_values
    ):
        """Test string trimming methods."""

        data = [{"val": v} for v in input_values]
        ds = _create_dataset(data, dataset_format)

        method = getattr(col("val").str, method_name)
        result = ds.with_column("result", method(*method_args)).to_pandas()

        expected = pd.DataFrame({"val": input_values, "result": expected_values})
        assert_df_equal(result, expected)


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
@pytest.mark.parametrize(
    "method_name,method_kwargs,expected_value",
    [
        ("pad", {"width": 5, "fillchar": "*", "side": "right"}, "hi***"),
        ("pad", {"width": 5, "fillchar": "*", "side": "left"}, "***hi"),
        ("pad", {"width": 6, "fillchar": "*", "side": "both"}, "**hi**"),
        ("center", {"width": 6, "padding": "*"}, "**hi**"),
    ],
)
class TestStringPadding:
    """Tests for string padding operations."""

    def test_string_padding(
        self, dataset_format, method_name, method_kwargs, expected_value
    ):
        """Test string padding methods."""

        data = [{"val": "hi"}]
        ds = _create_dataset(data, dataset_format)

        method = getattr(col("val").str, method_name)
        result = ds.with_column("result", method(**method_kwargs)).to_pandas()

        expected = pd.DataFrame({"val": ["hi"], "result": [expected_value]})
        assert_df_equal(result, expected)


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
@pytest.mark.parametrize(
    "method_name,method_args,method_kwargs,input_values,expected_results",
    [
        ("starts_with", ("A",), {}, ["Alice", "Bob", "Alex"], [True, False, True]),
        ("starts_with", ("A",), {"ignore_case": True}, ["alice", "bob"], [True, False]),
        ("ends_with", ("e",), {}, ["Alice", "Bob"], [True, False]),
        ("contains", ("li",), {}, ["Alice", "Bob", "Charlie"], [True, False, True]),
        ("find", ("i",), {}, ["Alice", "Bob"], [2, -1]),
        ("count", ("a",), {}, ["banana", "apple"], [3, 1]),
        ("match", ("Al%",), {}, ["Alice", "Bob", "Alex"], [True, False, True]),
    ],
)
class TestStringSearch:
    """Tests for string searching operations."""

    def test_string_search(
        self,
        dataset_format,
        method_name,
        method_args,
        method_kwargs,
        input_values,
        expected_results,
    ):
        """Test string searching methods."""

        data = [{"val": v} for v in input_values]
        ds = _create_dataset(data, dataset_format)

        method = getattr(col("val").str, method_name)
        result = ds.with_column(
            "result", method(*method_args, **method_kwargs)
        ).to_pandas()

        expected = pd.DataFrame({"val": input_values, "result": expected_results})
        assert_df_equal(result, expected)


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestStringTransform:
    """Tests for string transformation operations."""

    def test_reverse(self, dataset_format):
        """Test str.reverse() reverses strings."""

        data = [{"val": "hello"}, {"val": "world"}]
        ds = _create_dataset(data, dataset_format)
        result = ds.with_column("rev", col("val").str.reverse()).to_pandas()
        expected = pd.DataFrame({"val": ["hello", "world"], "rev": ["olleh", "dlrow"]})
        assert_df_equal(result, expected)

    def test_slice(self, dataset_format):
        """Test str.slice() extracts substring."""

        data = [{"val": "hello"}]
        ds = _create_dataset(data, dataset_format)
        result = ds.with_column("sliced", col("val").str.slice(1, 4)).to_pandas()
        expected = pd.DataFrame({"val": ["hello"], "sliced": ["ell"]})
        assert_df_equal(result, expected)

    def test_replace(self, dataset_format):
        """Test str.replace() replaces substring."""

        data = [{"val": "hello world"}]
        ds = _create_dataset(data, dataset_format)
        result = ds.with_column(
            "replaced", col("val").str.replace("world", "universe")
        ).to_pandas()
        expected = pd.DataFrame(
            {"val": ["hello world"], "replaced": ["hello universe"]}
        )
        assert_df_equal(result, expected)

    def test_replace_with_max(self, dataset_format):
        """Test str.replace() with max_replacements."""

        data = [{"val": "aaa"}]
        ds = _create_dataset(data, dataset_format)
        result = ds.with_column(
            "replaced", col("val").str.replace("a", "X", max_replacements=2)
        ).to_pandas()
        expected = pd.DataFrame({"val": ["aaa"], "replaced": ["XXa"]})
        assert_df_equal(result, expected)

    def test_repeat(self, dataset_format):
        """Test str.repeat() repeats strings."""

        data = [{"val": "A"}]
        ds = _create_dataset(data, dataset_format)
        result = ds.with_column("repeated", col("val").str.repeat(3)).to_pandas()
        expected = pd.DataFrame({"val": ["A"], "repeated": ["AAA"]})
        assert_df_equal(result, expected)


# ──────────────────────────────────────
# Struct Namespace Tests
# ──────────────────────────────────────


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestStructNamespace:
    """Tests for struct namespace operations."""

    def test_struct_field(self, dataset_format):
        """Test struct.field() extracts field."""

        # Arrow table with explicit struct types
        arrow_table = pa.table(
            {
                "user": pa.array(
                    [
                        {"name": "Alice", "age": 30},
                        {"name": "Bob", "age": 25},
                    ],
                    type=pa.struct(
                        [
                            pa.field("name", pa.string()),
                            pa.field("age", pa.int32()),
                        ]
                    ),
                )
            }
        )
        # Items representation
        items_data = [
            {"user": {"name": "Alice", "age": 30}},
            {"user": {"name": "Bob", "age": 25}},
        ]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column("age", col("user").struct.field("age")).to_pandas()
        expected = pd.DataFrame(
            {
                "user": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
                "age": [30, 25],
            }
        )
        assert_df_equal(result, expected)

    def test_struct_bracket(self, dataset_format):
        """Test struct['field'] bracket notation."""

        arrow_table = pa.table(
            {
                "user": pa.array(
                    [
                        {"name": "Alice", "age": 30},
                        {"name": "Bob", "age": 25},
                    ],
                    type=pa.struct(
                        [
                            pa.field("name", pa.string()),
                            pa.field("age", pa.int32()),
                        ]
                    ),
                )
            }
        )
        items_data = [
            {"user": {"name": "Alice", "age": 30}},
            {"user": {"name": "Bob", "age": 25}},
        ]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column("name", col("user").struct["name"]).to_pandas()
        expected = pd.DataFrame(
            {
                "user": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
                "name": ["Alice", "Bob"],
            }
        )
        assert_df_equal(result, expected)

    def test_struct_nested_field(self, dataset_format):
        """Test nested struct field access with .field()."""

        arrow_table = pa.table(
            {
                "user": pa.array(
                    [
                        {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
                        {"name": "Bob", "address": {"city": "LA", "zip": "90001"}},
                    ],
                    type=pa.struct(
                        [
                            pa.field("name", pa.string()),
                            pa.field(
                                "address",
                                pa.struct(
                                    [
                                        pa.field("city", pa.string()),
                                        pa.field("zip", pa.string()),
                                    ]
                                ),
                            ),
                        ]
                    ),
                )
            }
        )
        items_data = [
            {"user": {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}}},
            {"user": {"name": "Bob", "address": {"city": "LA", "zip": "90001"}}},
        ]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column(
            "city", col("user").struct.field("address").struct.field("city")
        ).to_pandas()
        expected = pd.DataFrame(
            {
                "user": [
                    {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
                    {"name": "Bob", "address": {"city": "LA", "zip": "90001"}},
                ],
                "city": ["NYC", "LA"],
            }
        )
        assert_df_equal(result, expected)

    def test_struct_nested_bracket(self, dataset_format):
        """Test nested struct field access with brackets."""

        arrow_table = pa.table(
            {
                "user": pa.array(
                    [
                        {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
                        {"name": "Bob", "address": {"city": "LA", "zip": "90001"}},
                    ],
                    type=pa.struct(
                        [
                            pa.field("name", pa.string()),
                            pa.field(
                                "address",
                                pa.struct(
                                    [
                                        pa.field("city", pa.string()),
                                        pa.field("zip", pa.string()),
                                    ]
                                ),
                            ),
                        ]
                    ),
                )
            }
        )
        items_data = [
            {"user": {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}}},
            {"user": {"name": "Bob", "address": {"city": "LA", "zip": "90001"}}},
        ]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column(
            "zip", col("user").struct["address"].struct["zip"]
        ).to_pandas()
        expected = pd.DataFrame(
            {
                "user": [
                    {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
                    {"name": "Bob", "address": {"city": "LA", "zip": "90001"}},
                ],
                "zip": ["10001", "90001"],
            }
        )
        assert_df_equal(result, expected)


# ──────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestNamespaceIntegration:
    """Tests for chaining and combining namespace expressions."""

    def test_list_with_arithmetic(self, dataset_format):
        """Test list operations combined with arithmetic."""

        data = [{"items": [1, 2, 3]}]
        ds = _create_dataset(data, dataset_format)
        result = ds.with_column("len_plus_one", col("items").list.len() + 1).to_pandas()
        expected = pd.DataFrame({"items": [[1, 2, 3]], "len_plus_one": [4]})
        assert_df_equal(result, expected)

    def test_string_with_comparison(self, dataset_format):
        """Test string operations combined with comparison."""

        data = [{"name": "Alice"}, {"name": "Bo"}]
        ds = _create_dataset(data, dataset_format)
        result = ds.with_column("long_name", col("name").str.len() > 3).to_pandas()
        expected = pd.DataFrame({"name": ["Alice", "Bo"], "long_name": [True, False]})
        assert_df_equal(result, expected)

    def test_multiple_operations(self, dataset_format):
        """Test multiple namespace operations in single pipeline."""

        data = [{"name": "alice"}]
        ds = _create_dataset(data, dataset_format)
        result = (
            ds.with_column("upper", col("name").str.upper())
            .with_column("len", col("name").str.len())
            .with_column("starts_a", col("name").str.starts_with("a"))
            .to_pandas()
        )
        expected = pd.DataFrame(
            {
                "name": ["alice"],
                "upper": ["ALICE"],
                "len": [5],
                "starts_a": [True],
            }
        )
        assert_df_equal(result, expected)


# ──────────────────────────────────────
# Datetime Namespace Tests
# ──────────────────────────────────────


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestDatetimeNamespace:
    """Tests for datetime namespace operations."""

    def test_dt_year(self, dataset_format):
        """Test dt.year() extracts year from timestamp."""

        arrow_table = pa.table(
            {
                "timestamp": pa.array(
                    ["2021-01-15 12:30:45", "2022-06-20 08:15:30"],
                    type=pa.timestamp("us"),
                )
            }
        )
        items_data = [
            {"timestamp": "2021-01-15 12:30:45"},
            {"timestamp": "2022-06-20 08:15:30"},
        ]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column("year", col("timestamp").dt.year()).to_pandas()
        assert result["year"].tolist() == [2021, 2022]

    def test_dt_month(self, dataset_format):
        """Test dt.month() extracts month from timestamp."""

        arrow_table = pa.table(
            {
                "timestamp": pa.array(
                    ["2021-01-15", "2021-12-20"], type=pa.timestamp("us")
                )
            }
        )
        items_data = [{"timestamp": "2021-01-15"}, {"timestamp": "2021-12-20"}]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column("month", col("timestamp").dt.month()).to_pandas()
        assert result["month"].tolist() == [1, 12]

    def test_dt_day(self, dataset_format):
        """Test dt.day() extracts day from timestamp."""

        arrow_table = pa.table(
            {
                "timestamp": pa.array(
                    ["2021-01-05", "2021-01-25"], type=pa.timestamp("us")
                )
            }
        )
        items_data = [{"timestamp": "2021-01-05"}, {"timestamp": "2021-01-25"}]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column("day", col("timestamp").dt.day()).to_pandas()
        assert result["day"].tolist() == [5, 25]

    def test_dt_hour(self, dataset_format):
        """Test dt.hour() extracts hour from timestamp."""

        arrow_table = pa.table(
            {
                "timestamp": pa.array(
                    ["2021-01-01 08:30:00", "2021-01-01 15:45:00"],
                    type=pa.timestamp("us"),
                )
            }
        )
        items_data = [
            {"timestamp": "2021-01-01 08:30:00"},
            {"timestamp": "2021-01-01 15:45:00"},
        ]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column("hour", col("timestamp").dt.hour()).to_pandas()
        assert result["hour"].tolist() == [8, 15]

    def test_dt_strftime(self, dataset_format):
        """Test dt.strftime() formats timestamp as string."""

        arrow_table = pa.table(
            {
                "timestamp": pa.array(
                    ["2021-01-15 12:30:45", "2022-06-20 08:15:30"],
                    type=pa.timestamp("us"),
                )
            }
        )
        items_data = [
            {"timestamp": "2021-01-15 12:30:45"},
            {"timestamp": "2022-06-20 08:15:30"},
        ]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column(
            "formatted", col("timestamp").dt.strftime("%Y-%m-%d")
        ).to_pandas()
        assert result["formatted"].tolist() == ["2021-01-15", "2022-06-20"]


# ──────────────────────────────────────
# Map Namespace Tests
# ──────────────────────────────────────


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestMapNamespace:
    """Tests for map namespace operations."""

    def test_map_keys(self, dataset_format):
        """Test map.keys() extracts keys from map."""

        arrow_table = pa.table(
            {
                "data": pa.array(
                    [[("a", 1), ("b", 2)], [("x", 10), ("y", 20)]],
                    type=pa.map_(pa.string(), pa.int32()),
                )
            }
        )
        # For pandas, we'll use arrow table directly
        if dataset_format == "arrow":
            ds = ray.data.from_arrow(arrow_table)
        else:
            # Skip pandas for map tests as pandas doesn't support map types well
            pytest.skip("Map type not fully supported in pandas format")

        result = ds.with_column("keys", col("data").map.keys()).to_pandas()
        # Keys should be lists of strings
        assert result["keys"].tolist() == [["a", "b"], ["x", "y"]]

    def test_map_values(self, dataset_format):
        """Test map.values() extracts values from map."""

        arrow_table = pa.table(
            {
                "data": pa.array(
                    [[("a", 1), ("b", 2)], [("x", 10), ("y", 20)]],
                    type=pa.map_(pa.string(), pa.int32()),
                )
            }
        )
        if dataset_format == "arrow":
            ds = ray.data.from_arrow(arrow_table)
        else:
            pytest.skip("Map type not fully supported in pandas format")

        result = ds.with_column("values", col("data").map.values()).to_pandas()
        # Values should be lists of integers
        assert result["values"].tolist() == [[1, 2], [10, 20]]


# ──────────────────────────────────────
# Array (Fixed-size) Namespace Tests
# ──────────────────────────────────────


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestArrayNamespace:
    """Tests for fixed-size array namespace operations."""

    def test_arr_flatten(self, dataset_format):
        """Test arr.flatten() converts fixed-size array to list."""

        arrow_table = pa.table(
            {
                "embeddings": pa.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    type=pa.list_(pa.float64(), 3),
                )
            }
        )
        if dataset_format == "arrow":
            ds = ray.data.from_arrow(arrow_table)
        else:
            pytest.skip("Fixed-size list type not well supported in pandas format")

        result = ds.with_column(
            "as_list", col("embeddings").arr.flatten()
        ).to_pandas()
        # Should convert to variable-length list
        assert result["as_list"].tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_arr_to_list(self, dataset_format):
        """Test arr.to_list() is an alias for flatten."""

        arrow_table = pa.table(
            {
                "embeddings": pa.array(
                    [[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float64(), 2)
                )
            }
        )
        if dataset_format == "arrow":
            ds = ray.data.from_arrow(arrow_table)
        else:
            pytest.skip("Fixed-size list type not well supported in pandas format")

        result = ds.with_column("as_list", col("embeddings").arr.to_list()).to_pandas()
        assert result["as_list"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


# ──────────────────────────────────────
# Direct Expr Methods Tests
# ──────────────────────────────────────


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestExprArithmeticMethods:
    """Tests for direct arithmetic methods on Expr."""

    def test_negate(self, dataset_format):
        """Test negate() method."""

        data = [{"value": 5}, {"value": -3}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("negated", col("value").negate()).to_pandas()
        assert result["negated"].tolist() == [-5, 3]

    def test_abs(self, dataset_format):
        """Test abs() method."""

        data = [{"value": -5}, {"value": 3}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("absolute", col("value").abs()).to_pandas()
        assert result["absolute"].tolist() == [5, 3]

    def test_sign(self, dataset_format):
        """Test sign() method."""

        data = [{"value": -5}, {"value": 0}, {"value": 3}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("sign_val", col("value").sign()).to_pandas()
        assert result["sign_val"].tolist() == [-1, 0, 1]


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestExprRoundingMethods:
    """Tests for rounding methods on Expr."""

    def test_ceil(self, dataset_format):
        """Test ceil() method."""

        data = [{"value": 2.3}, {"value": -2.7}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("ceiled", col("value").ceil()).to_pandas()
        assert result["ceiled"].tolist() == [3.0, -2.0]

    def test_floor(self, dataset_format):
        """Test floor() method."""

        data = [{"value": 2.7}, {"value": -2.3}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("floored", col("value").floor()).to_pandas()
        assert result["floored"].tolist() == [2.0, -3.0]

    def test_round(self, dataset_format):
        """Test round() method."""

        data = [{"value": 2.456}, {"value": 2.454}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("rounded", col("value").round(ndigits=2)).to_pandas()
        assert result["rounded"].tolist() == [2.46, 2.45]

    def test_trunc(self, dataset_format):
        """Test trunc() method."""

        data = [{"value": 2.7}, {"value": -2.7}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("truncated", col("value").trunc()).to_pandas()
        assert result["truncated"].tolist() == [2.0, -2.0]


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestExprLogarithmicMethods:
    """Tests for logarithmic methods on Expr."""

    def test_ln(self, dataset_format):
        """Test ln() method."""

        data = [{"value": 1.0}, {"value": 2.718281828459045}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("log_natural", col("value").ln()).to_pandas()
        # ln(1) = 0, ln(e) ≈ 1
        assert abs(result["log_natural"].tolist()[0] - 0.0) < 0.0001
        assert abs(result["log_natural"].tolist()[1] - 1.0) < 0.0001

    def test_log10(self, dataset_format):
        """Test log10() method."""

        data = [{"value": 1.0}, {"value": 100.0}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("log_10", col("value").log10()).to_pandas()
        assert result["log_10"].tolist() == [0.0, 2.0]

    def test_log2(self, dataset_format):
        """Test log2() method."""

        data = [{"value": 1.0}, {"value": 8.0}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("log_2", col("value").log2()).to_pandas()
        assert result["log_2"].tolist() == [0.0, 3.0]

    def test_exp(self, dataset_format):
        """Test exp() method."""

        data = [{"value": 0.0}, {"value": 1.0}]
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("exponential", col("value").exp()).to_pandas()
        # exp(0) = 1, exp(1) ≈ e
        assert abs(result["exponential"].tolist()[0] - 1.0) < 0.0001
        assert abs(result["exponential"].tolist()[1] - 2.718281828459045) < 0.0001


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestExprTrigonometricMethods:
    """Tests for trigonometric methods on Expr."""

    def test_sin(self, dataset_format):
        """Test sin() method."""

        data = [{"value": 0.0}, {"value": 1.5707963267948966}]  # 0 and π/2
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("sine", col("value").sin()).to_pandas()
        # sin(0) = 0, sin(π/2) = 1
        assert abs(result["sine"].tolist()[0] - 0.0) < 0.0001
        assert abs(result["sine"].tolist()[1] - 1.0) < 0.0001

    def test_cos(self, dataset_format):
        """Test cos() method."""

        data = [{"value": 0.0}, {"value": 3.141592653589793}]  # 0 and π
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("cosine", col("value").cos()).to_pandas()
        # cos(0) = 1, cos(π) = -1
        assert abs(result["cosine"].tolist()[0] - 1.0) < 0.0001
        assert abs(result["cosine"].tolist()[1] - (-1.0)) < 0.0001

    def test_tan(self, dataset_format):
        """Test tan() method."""

        data = [{"value": 0.0}, {"value": 0.7853981633974483}]  # 0 and π/4
        ds = _create_dataset(data, dataset_format)

        result = ds.with_column("tangent", col("value").tan()).to_pandas()
        # tan(0) = 0, tan(π/4) = 1
        assert abs(result["tangent"].tolist()[0] - 0.0) < 0.0001
        assert abs(result["tangent"].tolist()[1] - 1.0) < 0.0001


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestExprNullHandlingMethods:
    """Tests for null handling methods on Expr."""

    def test_fill_null(self, dataset_format):
        """Test fill_null() method."""

        arrow_table = pa.table({"value": pa.array([1, None, 3], type=pa.int32())})
        items_data = [{"value": 1}, {"value": None}, {"value": 3}]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column("filled", col("value").fill_null(-1)).to_pandas()
        assert result["filled"].tolist() == [1, -1, 3]

    def test_is_finite(self, dataset_format):
        """Test is_finite() method."""

        import numpy as np

        arrow_table = pa.table(
            {"value": pa.array([1.0, float("inf"), float("nan")], type=pa.float64())}
        )
        items_data = [{"value": 1.0}, {"value": np.inf}, {"value": np.nan}]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column("finite", col("value").is_finite()).to_pandas()
        assert result["finite"].tolist() == [True, False, False]

    def test_is_inf(self, dataset_format):
        """Test is_inf() method."""

        import numpy as np

        arrow_table = pa.table(
            {"value": pa.array([1.0, float("inf"), float("nan")], type=pa.float64())}
        )
        items_data = [{"value": 1.0}, {"value": np.inf}, {"value": np.nan}]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column("infinite", col("value").is_inf()).to_pandas()
        assert result["infinite"].tolist() == [False, True, False]

    def test_is_nan(self, dataset_format):
        """Test is_nan() method."""

        import numpy as np

        arrow_table = pa.table(
            {"value": pa.array([1.0, float("inf"), float("nan")], type=pa.float64())}
        )
        items_data = [{"value": 1.0}, {"value": np.inf}, {"value": np.nan}]
        ds = _create_dataset(items_data, dataset_format, arrow_table)

        result = ds.with_column("not_a_number", col("value").is_nan()).to_pandas()
        assert result["not_a_number"].tolist() == [False, False, True]


@pytest.mark.parametrize("dataset_format", DATASET_FORMATS)
class TestExprCastMethod:
    """Tests for cast method on Expr."""

    def test_cast_int_to_float(self, dataset_format):
        """Test cast() from int to float."""

        data = [{"value": 1}, {"value": 2}]
        ds = _create_dataset(data, dataset_format)

        from ray.data.datatype import DataType

        result = ds.with_column(
            "as_float", col("value").cast(DataType.float64())
        ).to_pandas()
        assert result["as_float"].dtype == "float64"
        assert result["as_float"].tolist() == [1.0, 2.0]

    def test_cast_float_to_int(self, dataset_format):
        """Test cast() from float to int."""

        data = [{"value": 1.5}, {"value": 2.7}]
        ds = _create_dataset(data, dataset_format)

        from ray.data.datatype import DataType

        result = ds.with_column(
            "as_int", col("value").cast(DataType.int32())
        ).to_pandas()
        assert result["as_int"].tolist() == [1, 2]


# ──────────────────────────────────────
# Error Handling Tests
# ──────────────────────────────────────


class TestNamespaceErrors:
    """Tests for proper error handling."""

    def test_list_invalid_index_type(self):
        """Test list bracket notation rejects invalid types."""

        with pytest.raises(TypeError, match="List indices must be integers or slices"):
            col("items").list["invalid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
