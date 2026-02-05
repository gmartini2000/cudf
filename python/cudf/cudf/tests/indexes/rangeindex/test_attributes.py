# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal

import inspect
import numpy as np
import cudf

def test_rangeindex_to_numpy_signature():
    sig = inspect.signature(cudf.RangeIndex.to_numpy)
    assert list(sig.parameters) == ["self", "dtype", "copy", "na_value"]
    assert sig.parameters["dtype"].default is None
    assert sig.parameters["copy"].default is False
    assert sig.parameters["na_value"].default is None

def test_rangeindex_to_numpy_caches_copy_false():
    idx = cudf.RangeIndex(0, 10, 1)
    a = idx.to_numpy(copy=False)
    b = idx.to_numpy(copy=False)
    assert a is b

def test_rangeindex_to_numpy_copy_true():
    idx = cudf.RangeIndex(0, 10, 1)
    a = idx.to_numpy(copy=False)
    b = idx.to_numpy(copy=True)
    assert a is not b
    assert np.array_equal(a, b)

def test_rangeindex_to_numpy_dtype():
    idx = cudf.RangeIndex(0, 10, 1)
    a = idx.to_numpy(dtype=np.int32, copy=False)
    assert a.dtype == np.int32

def test_rangeindex_to_numpy_accepts_na_value():
    idx = cudf.RangeIndex(0, 3, 1)
    a = idx.to_numpy(na_value=-1)
    assert a.tolist() == [0, 1, 2]

def test_rangeindex_values_host_routes_to_to_numpy_cache():
    idx = cudf.RangeIndex(0, 10, 1)
    a = idx.values_host
    b = idx.values_host
    assert a is b
    assert np.array_equal(a, idx.to_numpy(copy=False))


def test_rangeindex_contains():
    ridx = cudf.RangeIndex(start=0, stop=10, name="Index")
    assert 9 in ridx
    assert 10 not in ridx


@pytest.mark.parametrize(
    "start, stop, step", [(10, 20, 1), (0, -10, -1), (5, 5, 1)]
)
def test_range_index_is_unique_monotonic(start, stop, step):
    index = cudf.RangeIndex(start=start, stop=stop, step=step)
    index_pd = pd.RangeIndex(start=start, stop=stop, step=step)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize("data", [range(2), [10, 11, 12]])
def test_index_contains_hashable(data):
    gidx = cudf.Index(data)
    pidx = gidx.to_pandas()

    assert_exceptions_equal(
        lambda: [] in gidx,
        lambda: [] in pidx,
        lfunc_args_and_kwargs=((),),
        rfunc_args_and_kwargs=((),),
    )


def test_bool_rangeindex_raises():
    assert_exceptions_equal(
        lfunc=bool,
        rfunc=bool,
        lfunc_args_and_kwargs=[[pd.RangeIndex(0)]],
        rfunc_args_and_kwargs=[[cudf.RangeIndex(0)]],
    )


def test_from_pandas_rangeindex():
    idx1 = pd.RangeIndex(start=0, stop=4, step=1, name="myindex")
    idx2 = cudf.from_pandas(idx1)

    assert_eq(idx1.values, idx2.values)
    assert idx1.name == idx2.name


def test_rangeindex_constructor():
    gidx = cudf.RangeIndex(10)

    assert gidx._constructor is cudf.RangeIndex


def test_rangeindex_inferred_type():
    gidx = cudf.RangeIndex(10)
    pidx = pd.RangeIndex(10)
    assert_eq(gidx.inferred_type, pidx.inferred_type)