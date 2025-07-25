import numpy as np
import pytest

from cardiokoop.utils import (
    stack_data,
    stack_data_with_control,
    num_shifts_in_stack,
    set_defaults,
)


def test_stack_data_basic():
    # Two trajectories of length 3; with 1 shift, new_len_time = 2
    data = np.arange(6)[:, None]
    tensor = stack_data(data, num_shifts=1, len_time=3)
    assert tensor.shape == (2, 4, 1)
    # First slice: rows [0,1] and [3,4]; second slice: [1,2] and [4,5]
    assert np.array_equal(tensor[0, :, 0], [0, 1, 3, 4])
    assert np.array_equal(tensor[1, :, 0], [1, 2, 4, 5])


def test_stack_data_invalid():
    # Scalars / 0D arrays should raise
    with pytest.raises(ValueError):
        stack_data(np.array(5), num_shifts=1, len_time=3)


def test_num_shifts_in_stack():
    params = {'num_shifts': 3, 'num_shifts_middle': 2}
    assert num_shifts_in_stack(params) == 3
    params = {'num_shifts': 0, 'num_shifts_middle': 4}
    assert num_shifts_in_stack(params) == 4


def test_set_defaults_minimal():
    params = {}
    set_defaults(params)
    # Default flags for progress checks should be present and False
    assert all(flag in params and not params[flag]
               for flag in ['been5min', 'been20min', 'been40min',
                            'been1hr', 'been2hr', 'been3hr', 'been4hr', 'beenHalf'])