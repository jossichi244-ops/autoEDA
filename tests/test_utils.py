import pytest
import pandas as pd
import numpy as np
from app.main import (
    convert_numpy_types,
    infer_schema_from_df,
    analyze_column,
    clean_column_name,
    safe_float,
)

def test_convert_numpy_types():
    assert convert_numpy_types(np.int64(42)) == 42
    assert convert_numpy_types(np.float64(3.14)) == 3.14
    assert convert_numpy_types(np.nan) is None
    assert convert_numpy_types(pd.NA) is None
    assert convert_numpy_types([np.int32(1), np.float32(2.5)]) == [1, 2.5]
    assert convert_numpy_types({"a": np.bool_(True)}) == {"a": True}
    assert convert_numpy_types("hello") == "hello"

def test_infer_schema_from_df():
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.1, 2.2, 3.3],
        "bool_col": [True, False, True],
        "str_col": ["a", "b", "c"],
        "date_col": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    })
    schema = infer_schema_from_df(df)
    assert schema["properties"]["int_col"]["type"] == "integer"
    assert schema["properties"]["float_col"]["type"] == "number"
    assert schema["properties"]["bool_col"]["type"] == "boolean"
    assert schema["properties"]["str_col"]["type"] == "string"
    assert schema["properties"]["date_col"]["type"] == "string"

def test_analyze_column():
    series = pd.Series([1, 2, 3, None, 4], name="test_col")
    result = analyze_column(series)
    assert result["name"] == "test_col"
    assert result["non_null"] == 4
    assert result["nulls"] == 1
    assert result["inferred_type"] == "numeric"
    assert result["stats"]["min"] == 1
    assert result["stats"]["max"] == 4

def test_clean_column_name():
    assert clean_column_name("  User Name  ") == "user_name"
    assert clean_column_name("Price ($)") == "price_"
    assert clean_column_name("first_name__last_name") == "first_name_last_name"
    assert clean_column_name("email@domain.com") == "email_domain_com"

def test_safe_float():
    assert safe_float(None) is None
    assert safe_float(np.nan) is None
    assert safe_float(np.inf) is None
    assert safe_float("123.45") == 123.45
    assert safe_float(42) == 42
    assert safe_float("abc") is None