import numpy as np

import scipy.linalg as sla
from rttddft.propagators.propstate import PropagatorState

def step_magnus4(state, h1e, v_ext, S, get_veff, dt, conv_tol=1e-5, mo_basis=False, bc=None, logger=None, callback=None, **kwargs):
    """Fourth-order Magnus predictor/corrector step (J. Chem. Phys. 121, 3425-3433 (2004)).

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
    **kwargs : dict, optional
        Additional keyword arguments.
        hkin : np.ndarray
            Kinetic-energy integrals in the same basis as ``state.fock`` / ``v_ext``.
        v_ext_nl : np.ndarray or callable, optional
            Spatially nonlocal external potential (e.g. GTH PP) for the
            commutator ``[T + V_ext^{nl}, ΔV]``.
            If callable, invoked as ``v_ext_nl(t)``.
    Returns
    -------
    PropagatorState
        New system state after the time step.
    """
    hkin = kwargs.get('hkin')
    assert hkin is not None, "Kinetic integrals 'hkin' must be provided for 4th order Magnus"

    converged = False
    nbuilds = 0
    dm = state.dm
    F_t = state.fock
    F_m_dt = state.fock_prev
    t = state.time

    dm_p_dt = state.dm

    if dm.ndim > 2:
        nkpts = dm.shape[0]
        is_kpoint = True
    else:
        nkpts = 0
        is_kpoint = False
        if not mo_basis:
            Sinv = sla.inv(S)

    frac1 = 0.5 - np.sqrt(3)/6
    frac2 = 0.5 + np.sqrt(3)/6

    v_ext_t1 = v_ext(t + frac1 * dt)
    v_ext_t2 = v_ext(t + frac2 * dt)

    v_nl = kwargs.get('v_ext_nl', None)
    if callable(v_nl):
        v_nl = v_nl(t)
    if v_nl is None:
        T_left = hkin
    else:
        T_left = hkin + v_nl

    while not converged:
        if nbuilds == 0:
            # Linear extrapolation for F_t+dt
            dF = F_t - F_m_dt
        else:
            # Interpolate between t and t+dt using the current F_p_dt estimate
            dF = F_p_dt - F_t
        F_t1 = F_t + frac1 * dF
        F_t2 = F_t + frac2 * dF

        W1 = F_t1 + v_ext_t1
        W2 = F_t2 + v_ext_t2

        dV = np.sqrt(3) / 12.0 * dt * (W2 - W1)
        H_avg = (W1 + W2) / 2.0

        if is_kpoint:
            dm_p_dt_new = np.zeros_like(dm)
            for k in range(nkpts):
                if mo_basis:
                    commutator = T_left[k] @ dV[k] - dV[k] @ T_left[k]
                    H4_k = H_avg[k] + 1j * commutator
                    evs, evecs = sla.eigh(H4_k)
                    expH4_k = evecs @ (np.exp(-1.0j * dt * evs)[:, None] * evecs.conj().T)
                else:
                    Sinv = sla.inv(S[k])
                    commutator = T_left[k] @ Sinv @ dV[k] - dV[k] @ Sinv @ T_left[k]
                    H4_k = H_avg[k] + 1j * commutator
                    evs, C2 = sla.eigh(H4_k, b=S[k])
                    C2inv = sla.inv(C2)
                    expH4_k = C2 @ (np.exp(-1.0j * dt * evs)[:, None] * C2inv)
                dm_p_dt_new[k] = expH4_k @ dm[k] @ expH4_k.conj().T
        else:
            if mo_basis:
                commutator = T_left @ dV - dV @ T_left
                H4 = H_avg + 1j * commutator
                evs, evecs = sla.eigh(H4)
                expH4 = evecs @ (np.exp(-1.0j * dt * evs)[:, None] * evecs.conj().T)
            else:
                commutator = T_left @ Sinv @ dV - dV @ Sinv @ T_left
                H4 = H_avg + 1j * commutator
                evs, C2 = sla.eigh(H4, b=S)
                C2inv = sla.inv(C2)
                expH4 = C2 @ (np.exp(-1.0j * dt * evs)[:, None] * C2inv)
            dm_p_dt_new = expH4 @ dm @ expH4.conj().T

        diff = np.linalg.norm(dm_p_dt_new - dm_p_dt)
        dm_p_dt = dm_p_dt_new

        if mo_basis:
            assert bc is not None, "BasisChanger 'bc' must be provided for MO-basis Magnus4"
            dm_p_dt_ao = bc.rev_denslike(dm_p_dt)
            F_p_dt_ao = h1e + get_veff(dm=dm_p_dt_ao)
            F_p_dt = bc.rotate_focklike(F_p_dt_ao)
        else:
            F_p_dt = h1e + get_veff(dm=dm_p_dt)

        if diff < conv_tol:
            converged = True
        else:
            nbuilds += 1

    if logger is not None:
        logger.debug(f'Magnus4: time {t:.3f}, {nbuilds + 1} get_veff call(s), |drho| = {diff:1.3e}')

    new_state = PropagatorState(
        dm=dm_p_dt,
        dm_min_half=None,
        fock=F_p_dt,
        fock_prev=F_t,
        time=t + dt,
        time_prev=t
    )

    if callback:
        callback(new_state)
    return new_state
