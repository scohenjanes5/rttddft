import math
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf import hf_symm
from pyscf.scf import _response_functions
from pyscf.data import nist
from pyscf.tdscf.rhf import _charge_center

import h5py
import scipy.linalg as sla
from rttddft.propagators.propstate import PropagatorState
from rttddft.propagators import magnus2, mmut, magnus4
from rttddft.lib import BasisChanger

RTSCF_PROP_METHODS = {'magnus2': magnus2.step_magnus2, 'mmut': mmut.step_mmut, 'magnus4': magnus4.step_magnus4}

def apply_delta_kick_dm(dm, V, dt, S=None, mo_basis=True):
    """Apply a discrete delta kick: ρ → U ρ U† with U = exp(-i dt V).

    ``kick_field`` returns a peak field value sampled as a Hamiltonian term for
    one timestep (as in MMUT's ``exp(-i dt (F+V))``), so the analytic boost uses
    ``dt V`` rather than ``V`` alone.
    """
    def _kick_one(dm1, V1, S1):
        if mo_basis:
            U = sla.expm(-1.0j * dt * V1)
        else:
            assert S1 is not None, "Overlap S is required for AO delta kick"
            evs, C = sla.eigh(V1, b=S1)
            U = C @ (np.exp(-1.0j * dt * evs)[:, None] * (C.conj().T @ S1))
        return U @ dm1 @ U.conj().T

    if dm.ndim > 2:
        out = np.empty_like(dm, dtype=complex)
        for k in range(dm.shape[0]):
            Sk = None if S is None else S[k]
            out[k] = _kick_one(dm[k], V[k], Sk)
        return out
    return _kick_one(dm, V, S)


def maybe_apply_delta_kick(dm, v_ext, t_start, dt, S=None, mo_basis=True, logger=None,
                           v_ext_nl=None):
    """If ``v_ext(t_start) ≠ 0``, apply an instantaneous kick and consume that spike.

    Handles a delta at ``t_start`` alone or together with a later continuous drive:
    the density is kicked with ``U = exp(-i dt V(t_start))``, then ``v_ext`` is wrapped
    so that sample is not applied again during propagation.

    ``v_ext_nl``, when given, is treated as a static nonlocal contribution (e.g. GTH
    PP) and is excluded from the kick so it is not mistaken for an impulse.

    Returns
    -------
    dm, v_ext
    """
    V0 = v_ext(t_start)
    V_kick = V0 if v_ext_nl is None else V0 - v_ext_nl
    if np.linalg.norm(V_kick) == 0.0:
        return dm, v_ext
    if logger is not None:
        logger.info('Applying instantaneous delta kick at t=%g', t_start)
    dm = apply_delta_kick_dm(dm, V_kick, dt, S=S, mo_basis=mo_basis)

    def v_ext_consumed(t):
        Vt = v_ext(t)
        if np.isclose(t, t_start):
            return Vt - V_kick
        return Vt
    return dm, v_ext_consumed


def gpulse_efield(t0, peak, sigma, dir=(0,0,1.0), freq=0.0, phaseshift=0.0):
    """
    Gaussian pulse electric field

    Parameters:
    t0 (float): The center time of the pulse.
    peak (float): The peak amplitude of the pulse.
    sigma (float): The width of the pulse.
    dir (tuple, optional): The direction of the electric field. Defaults to (0, 0, 1.0).
    freq (float, optional): The frequency of the sinusoidal component of the pulse. Defaults to 0.0.
    phaseshift (float, optional): The phase shift of the sinusoidal component of the pulse. Defaults to 0.0.

    Returns:
    function: A function that calculates the electric field at a given time.
    """
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    
    def E(t):
        return (peak * np.exp( -(t-t0)**2 / (2*sigma**2) ) * \
                np.sin(freq*t + phaseshift)) * uhat
    return E

def kick_field(t0, peak, dir=(0,0,1.0)):
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    def E(t):
        return (np.isclose(t,t0) * peak) * uhat
    return E

def sine_efield(E0, omega, dir=(0,0,1.0), start=0.0, shift=0.0):
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    def E(t):
        if t < start:
            return 0.0 * uhat
        else:
            return E0 * np.sin(omega * (t - start) - shift) * uhat
    return E

def cos2_envelope(t, sigma):
    if np.abs(t-sigma)<=sigma:
        return np.cos(np.pi/(2*sigma) * (sigma-t))**2
    else:
        return 0.0

def cos2_ewave(E0, omega, sigma, dir=(0,0,1.0), start=0.0, shift=0.0):
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    def E(t):
        return E0 * np.sin(omega * (t - start) - shift) * cos2_envelope(t, sigma) * uhat
    return E

def intensity_to_e0_au(intensity):
    """Convert intensity in W/cm^2 to E0 in atomic units

    Parameters
    ----------
    intensity : float
        Intensity in W/cm^2
    
    Returns
    -------
    float
        E0 in atomic units
    """
    return np.sqrt(intensity / (3.50944758e16))

def make_bc(mf):
    return BasisChanger(mf.get_ovlp(), mf.mo_coeff)

def get_mo_dip(bc, mol, origin=None):
    if origin is None:
        origin = _charge_center(mol)
    with mol.with_common_origin(origin):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    return bc.rotate_focklike(ao_dip)

def make_vext_from_efield(efield, mo_dip):
    if efield is not None:
        v_ext = lambda t: -np.einsum('xij,x->ij', mo_dip, efield(t))
    else: 
        v_ext = lambda t: np.zeros_like(mo_dip[0])
    return v_ext

class RTTDSCF(lib.StreamObject):
    def __init__(self, mf, prop_method='magnus2', chkfile = None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory
        self.prop = None
        self.trace = None
        self.chkfile = chkfile

        self.wfnsym = None
        if prop_method not in RTSCF_PROP_METHODS:
            raise ValueError(f'prop_method {prop_method} not recognized')
        self.prop_method = prop_method

    
    def kernel(self, t_end, dt, t_start=0.0, efield=None, mo_basis=True):
        bc = BasisChanger(self._scf.get_ovlp(), self._scf.mo_coeff)
        log = logger.new_logger(self, self.verbose)

        with self.mol.with_common_origin(_charge_center(self.mol)):
            ao_dip = self.mol.intor_symmetric('int1e_r', comp=3)

        S = self._scf.get_ovlp()
        Sinv = sla.inv(S)
        mo_dip = bc.rotate_focklike(ao_dip)
        charges = self.mol.atom_charges()
        coords  = self.mol.atom_coords()
        nucl_dip = np.einsum('i,ix->x', charges, coords)

        self.trace = {'t': [], 'dipole': [], 'dm': []}

        if t_end <= t_start:
            raise ValueError('t_end must be greater than t_start')

        nsteps = math.ceil((t_end - t_start) / dt)

        nao = self.mol.nao
        chkf = h5py.File(self.chkfile, "w") if self.chkfile is not None else None
        if chkf is not None:
            chkf.create_dataset('t', (0,), maxshape=(None,), dtype=np.float64, chunks=True)
            chkf.create_dataset('dipole', (0, 3), maxshape=(None, 3), dtype=np.complex128, chunks=True)
            chkf.create_dataset('dm', (0, nao, nao),
                                dtype=np.complex128,
                                maxshape=(None, nao, nao),
                                chunks=(1, nao, nao))
            chkf.create_dataset('fock', (0, nao, nao),
                                dtype=np.complex128,
                                maxshape=(None, nao, nao),
                                chunks=(1, nao, nao))
            # Optional: fock_intermediates / dm_min_half created on first write.

        def stepcallback(state):
            t = state.time
            dm = state.dm
            if mo_basis:
                dipole = -np.sum(mo_dip * dm[None,:,:].real, axis=(1,2))
            else:
                dipole = -np.sum(ao_dip * dm[None,:,:].real, axis=(1,2))
            self.trace['t'].append(t)
            self.trace['dipole'].append(dipole)
            self.trace['dm'].append(dm.copy())
            if chkf is not None:
                chkf['t'].resize((chkf['t'].shape[0] + 1), axis=0)
                chkf['dipole'].resize((chkf['dipole'].shape[0] + 1), axis=0)
                chkf['dm'].resize((chkf['dm'].shape[0] + 1), axis=0)
                chkf['fock'].resize((chkf['fock'].shape[0] + 1), axis=0)
                chkf['t'][-1] = t
                chkf['dipole'][-1] = np.asarray(dipole, dtype=np.complex128)
                chkf['dm'][-1] = np.asarray(dm, dtype=np.complex128)
                chkf['fock'][-1] = np.asarray(state.fock, dtype=np.complex128)
                if state.fock_intermediates is not None:
                    fi = np.asarray(state.fock_intermediates, dtype=np.complex128)
                    if 'fock_intermediates' not in chkf:
                        n_inter = fi.shape[0]
                        chkf.create_dataset(
                            'fock_intermediates', (0, n_inter, nao, nao),
                            dtype=np.complex128,
                            maxshape=(None, n_inter, nao, nao),
                            chunks=(1, n_inter, nao, nao),
                        )
                    chkf['fock_intermediates'].resize(
                        (chkf['fock_intermediates'].shape[0] + 1), axis=0)
                    chkf['fock_intermediates'][-1] = fi
                if state.dm_min_half is not None:
                    dmh = np.asarray(state.dm_min_half, dtype=np.complex128)
                    if 'dm_min_half' not in chkf:
                        chkf.create_dataset(
                            'dm_min_half', (0, nao, nao),
                            dtype=np.complex128,
                            maxshape=(None, nao, nao),
                            chunks=(1, nao, nao),
                        )
                    chkf['dm_min_half'].resize((chkf['dm_min_half'].shape[0] + 1), axis=0)
                    chkf['dm_min_half'][-1] = dmh

        if self.prop is None:
            if self.prop_method in RTSCF_PROP_METHODS:
                self.prop = RTSCF_PROP_METHODS[self.prop_method]
            else:
                raise ValueError(f'prop_method {self.prop_method} not recognized')

        dm = self._scf.make_rdm1()
        hkin = self.mol.intor('int1e_kin')
        h1e = hkin + self.mol.intor('int1e_nuc')
        get_veff = self._scf.get_veff

        if mo_basis:
            v_ext = make_vext_from_efield(efield, mo_dip)
            dm = bc.rotate_denslike(dm)
            hkin_prop = bc.rotate_focklike(hkin)
            dm, v_ext = maybe_apply_delta_kick(dm, v_ext, t_start, dt, mo_basis=True, logger=log)
            fock_init = bc.rotate_focklike(h1e + get_veff(dm=bc.rev_denslike(dm)))
        else:
            v_ext = make_vext_from_efield(efield, ao_dip)
            hkin_prop = hkin
            dm, v_ext = maybe_apply_delta_kick(dm, v_ext, t_start, dt, S=S, mo_basis=False, logger=log)
            fock_init = h1e + get_veff(dm=dm)

        prop_state = PropagatorState(
                    dm = dm,
                    dm_min_half = dm,
                    fock = fock_init,
                    fock_prev = fock_init,
                    time = t_start,
                    time_prev = t_start
                    )

        for _ in range(nsteps):
            prop_state = self.prop(
                state = prop_state,
                h1e = h1e,
                v_ext = v_ext,
                S = S,
                get_veff = get_veff,
                dt = dt,
                conv_tol = 1e-5,
                mo_basis = mo_basis,
                bc = bc,
                logger = log,
                callback = stepcallback,
                hkin = hkin_prop,
                Sinv = Sinv
            )
