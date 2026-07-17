import numpy as np

import scipy.linalg as sla
from rttddft.propagators.propstate import PropagatorState

def step_magnus2(state, h1e, v_ext, S, get_veff, dt, conv_tol=1e-5, mo_basis=False, bc=None, logger=None, callback=None, **kwargs):
    """Perform a single predictor/corrector time step using the Magnus expansion.

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
        Convergence tolerance for the predictor/corrector step.
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
    converged = False
    nbuilds = 0
    dm = state.dm
    F = state.fock
    F_m_dt = state.fock_prev
    F_p_half = 1.5 * F - 0.5 * F_m_dt

    t = state.time

    dm_p_dt = state.dm
    F_p_dt = F_p_half

    if dm.ndim > 2:
        nkpts = dm.shape[0]
        is_kpoint = True
    else:
        nkpts = 0
        is_kpoint = False

    v_ext_half = v_ext(t + 0.5 * dt)

    while not converged:

        W = (F_p_half + v_ext_half)
        # k-point case
        # todo: MPI parallelization
        if is_kpoint:
            dm_p_dt_new = np.zeros_like(dm)
            for k in range(nkpts):
                if mo_basis:
                    evs, evecs = sla.eigh(W[k])
                    expw_k = evecs @ (np.exp(-1.0j * dt * evs)[:, None] * evecs.conj().T)
                else:
                    evs, C2 = sla.eigh(W[k], b=S[k])
                    C2inv = sla.inv(C2)
                    expw_k = C2 @ (np.exp(-1.0j * dt * evs)[:, None] * C2inv)
                dm_p_dt_new[k] = expw_k @ dm[k] @ expw_k.conj().T

        else:
            if mo_basis:
                evs, evecs = sla.eigh(W)
                expw = evecs @ (np.exp(-1.0j * dt * evs)[:, None] * evecs.conj().T)
            else:
                evs, C2 = sla.eigh(W, b=S)
                C2inv = sla.inv(C2)
                expw = C2 @ (np.exp(-1.0j * dt * evs)[:, None] * C2inv)
            dm_p_dt_new = expw @ dm @ expw.conj().T

        diff = np.linalg.norm(dm_p_dt_new - dm_p_dt)

        dm_p_dt = dm_p_dt_new

        if diff < conv_tol:
            converged = True
        else:
            if mo_basis:
                assert bc is not None, "BasisChanger 'bc' must be provided to define the MO basis"
                dm_p_dt_ao = bc.rev_denslike(dm_p_dt)
                F_p_dt_ao = h1e + get_veff(dm=dm_p_dt_ao)
                F_p_dt = bc.rotate_focklike(F_p_dt_ao)
            else:
                F_p_dt = h1e + get_veff(dm=dm_p_dt)

            nbuilds += 1
            F_p_half = 0.5 * (F + F_p_dt)

    if logger is not None:
        logger.debug(f'Magnus2: time {t:.3f}, {nbuilds} get_veff call(s), |drho| = {diff:1.3e}')

    new_state = PropagatorState(
        dm=dm_p_dt,
        dm_min_half=None,
        fock=F_p_dt,
        fock_prev=F,
        time=t + dt,
        time_prev=t,
        fock_intermediates=np.asarray(F_p_half)[None, ...],
    )

    if callback:
        callback(new_state)
    return new_state
