import numpy as np

import scipy.linalg as sla
from rttddft.propagators.propstate import PropagatorState

def step_mmut(state, h1e, v_ext, S, get_veff, dt, conv_tol=1e-5, mo_basis=False, bc=None, logger=None, callback=None, **kwargs):
    """Perform a single time step with MMUT.

    Parameters
    ----------
    state : PropagatorState
        Current system state.
    h1e : np.ndarray
        Time-independent part of the one-electron Hamiltonian.
    v_ext : function
        Function returning the external potential at a given time.
    get_veff : function
        Function mapping the density matrix to the effective potential.
    dt : float
        Time step length
    conv_tol : float, by default 1e-5
        Not used.
    mo_basis : bool, optional
        Whether to use the molecular orbital basis, by default False
    bc : BasisChanger, optional
        basis changer for MO basis; only needed if mo_basis is True, by default None
    logger : pyscf.lib.logger, optional
        logger object, by default None
    callback : function, optional
        function to call after each time step, by default None.
        Invoked as callback(new_state), where new_state is of type PropagatorState.

    Returns
    -------
    PropagatorState
        New system state after the time step.
    """
    F = state.fock
    dm = state.dm
    nbuilds = 0
    t = state.time

    if dm.ndim > 2:
        nkpts = dm.shape[0]
        is_kpoint = True
    else:
        nkpts = 0
        is_kpoint = False

    W = F + v_ext(t)

    if not is_kpoint:
        if mo_basis:
            evs, evecs = sla.eigh(W)
            expw = evecs @ (np.exp(-1.0j * dt * evs)[:, None] * evecs.conj().T)
            expw_half = evecs @ (np.exp(-0.5j * dt * evs)[:, None] * evecs.conj().T)
        else:
            evs, C2 = sla.eigh(W, b=S)
            C2inv = sla.inv(C2)
            expw = C2 @ (np.exp(-1.0j * dt * evs)[:, None] * C2inv)
            expw_half = C2 @ (np.exp(-0.5j * dt * evs)[:, None] * C2inv)
    else:
        expw = np.zeros_like(W)
        expw_half = np.zeros_like(W)
        for k in range(nkpts):
            if mo_basis:
                evs, evecs = sla.eigh(W[k])
                expw[k] = evecs @ (np.exp(-1.0j * dt * evs)[:, None] * evecs.conj().T)
                expw_half[k] = evecs @ (np.exp(-0.5j * dt * evs)[:, None] * evecs.conj().T)
            else:
                evs, C2 = sla.eigh(W[k], b=S[k])
                C2inv = sla.inv(C2)
                expw[k] = C2 @ (np.exp(-1.0j * dt * evs)[:, None] * C2inv)
                expw_half[k] = C2 @ (np.exp(-0.5j * dt * evs)[:, None] * C2inv)

    if state.dm_min_half is None:
        dm_min_half = state.dm
    else:
        dm_min_half = state.dm_min_half

    # Should work for both k-point and non-k-point cases.
    if not is_kpoint:
        dm_p_half = expw @ dm_min_half @ expw.conj().T
        dm_p_dt = expw_half @ dm_p_half @ expw_half.conj().T
    else:
        dm_p_half = np.zeros_like(dm)
        dm_p_dt = np.zeros_like(dm)
        for k in range(nkpts):
            dm_p_half[k] = expw[k] @ dm_min_half[k] @ expw[k].conj().T
            dm_p_dt[k] = expw_half[k] @ dm_p_half[k] @ expw_half[k].conj().T


    if mo_basis:
        assert bc is not None, "BasisChanger 'bc' must be provided to define the MO basis"
        dm_p_dt_ao = bc.rev_denslike(dm_p_dt)
        F_p_dt_ao = h1e + get_veff(dm=dm_p_dt_ao)
        F_p_dt = bc.rotate_focklike(F_p_dt_ao)
    else:
        F_p_dt = h1e + get_veff(dm=dm_p_dt)

    nbuilds += 1

    if logger is not None:
        logger.debug(f'MMUT: time {t+dt:.3f}')


    new_state = PropagatorState(
        dm=dm_p_dt,
        dm_min_half=dm_p_half,
        fock=F_p_dt,
        fock_prev=F,
        time=t + dt,
        time_prev=t
    )

    if callback:
        callback(new_state)
    return new_state

