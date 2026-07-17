import typing
import numpy as np

class PropagatorState(typing.NamedTuple):
    dm: np.ndarray
    dm_min_half: typing.Optional[np.ndarray]
    fock: np.ndarray
    fock_prev: np.ndarray
    time: float
    time_prev: float
    fock_intermediates: typing.Optional[np.ndarray] = None
