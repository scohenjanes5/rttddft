import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.pbc import df as pbcdf
from pyscf.pbc.gto import pseudo

from pyscf.pbc.df import gdf_builder, aft, rsdf_builder
from pyscf.pbc.df import rsdf
from pyscf.pbc.gto.pseudo.ppnl_velgauge import VelGaugePPNLHelper, get_gth_pp_nl_velgauge, get_gth_pp_nl_velgauge_commutator
from pyscf import __config__


import math
import scipy

from pyscf.data import nist

from rttddft import rttdbase
from rttddft.rttdbase import make_vext_from_efield, get_mo_dip
from rttddft.propagators.propstate import PropagatorState
from rttddft.propagators import magnus2, mmut
from rttddft.lib import BasisChanger, KBasisChanger

RTSCF_PROP_METHODS = rttdbase.RTSCF_PROP_METHODS

import h5py

def kick_afield(t0, peak, dir=(0,0,1.0)):
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    def E(t):
        return (np.sign(t-t0) * peak) * uhat
    return E

def gaussian_afield(t0, sigma, peak, dir=(0,0,1.0)):
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    def E(t):
        return scipy.special.erf((t - t0) / sigma) * peak * uhat
    return E

def get_pseudopotential_local_part(mf, kpts=None):
    """Get the local part of the pseudopotential.
    Relevant for velocity-gauge calculations, where
    the nonlocal part of the pseudopotential depends on the vector potential
    and is frequently updated during time propagation.

    Works for AFT, GDF, and RSDF.

    Parameters
    ----------
    mf : PBC SCF object
    kpts : array_like, optional
        k-points, by default None

    Returns
    -------
    np.ndarray
        Local part of the pseudopotential.
    """
    cell = mf.cell
    if not cell.pseudo:
        raise RuntimeError('get_pseudopotential_local_part only works for pseudopotential calculations.')
    if mf.with_df is None:
        raise RuntimeError('get_pseudopotential_local_part only works for DF calculations.')
    with_df = mf.with_df
    cell = with_df.cell
    kpts, is_single_kpt = aft._check_kpts(with_df, kpts)

    if isinstance(with_df, pbcdf.GDF) or isinstance(mf.with_df, rsdf.RSDF):
        if with_df._prefer_ccdf or cell.omega > 0:
            # For long-range integrals _CCGDFBuilder is the only option
            nuc_builder = gdf_builder._CCNucBuilder(cell, kpts).build()
        else:
            nuc_builder = rsdf_builder._RSNucBuilder(cell, kpts).build()
        t0 = (logger.process_clock(), logger.perf_counter())
        vpp_loc_part1 = nuc_builder.get_pp_loc_part1()
        t1 = logger.timer_debug1(nuc_builder, 'get_pp_loc_part1', *t0)
    elif isinstance(mf.with_df, aft.AFTDF):
        nuc_builder = with_df
        t0 = (logger.process_clock(), logger.perf_counter())
        vpp_loc_part1 = aft._get_pp_loc_part1(nuc_builder, kpts, with_pseudo=True)
        t1 = logger.timer_debug1(with_df, 'get_pp_loc_part1', *t0)
    else:
        raise NotImplementedError
    pp2builder = aft._IntPPBuilder(cell, kpts)
    vpp_loc_part_2 = pp2builder.get_pp_loc_part2()
    t2 = logger.timer_debug1(nuc_builder, 'get_pp_loc_part2', *t1)
    vpp_loc = vpp_loc_part1 + vpp_loc_part_2
    if is_single_kpt:
        vpp_loc = vpp_loc[0]
    return vpp_loc

def make_vext_velgauge(cell, afield, kpts, h1e_ipovlp, bc=None, vgppnl_helper=None):
    def v_ext(t):
        # q is -1 for electrons
        qA = -afield(t)
        qA_dot_p = np.einsum('i,kixy->kxy', qA, h1e_ipovlp) * (1.0j)
        qA_sqr = np.dot(qA, qA)
        if cell.pseudo:
            pp_nl = get_gth_pp_nl_velgauge(cell, qA*0, kpts=kpts, vgppnl_helper=vgppnl_helper)
        else:
            pp_nl = 0.0
        nao = cell.nao_nr()
        vext_ao = (qA_sqr * np.eye(nao))[None, :, :] - 2.0 * qA_dot_p + pp_nl
        if bc is not None:
            vext_mo = bc.rotate_focklike(vext_ao)
            return vext_mo
        return vext_ao
    return v_ext

def get_electronic_velocity(cell, A, kpts, h1e_ipovlp, bc=None, dm=None, vgppnl_helper=None):
    qA = -1.0 * A
    if cell.pseudo:
        r_vnl_commutator = get_gth_pp_nl_velgauge_commutator(cell, qA, kpts=kpts, vgppnl_helper=vgppnl_helper)
    velocity = np.zeros(3, dtype=np.complex128)
    for k in range(len(kpts)):
        velocity += np.einsum('ixy,xy->i', h1e_ipovlp[k], dm[k]) * (1.0j)
        if cell.pseudo:
            velocity += np.einsum('ixy,xy->i', r_vnl_commutator[k], dm[k]) / (1.0j)
        velocity -= qA * np.trace(dm[k])
    return velocity

class KRTTDSCF(rttdbase.RTTDSCF):
    _keys = {'cell', 'h1e_nuc_local', 'h1e_kin', 'h1e_ipovlp', 'vgppnl_helper'}

    def __init__(self, mf, prop_method='magnus2', chkfile = None):
        super().__init__(mf, prop_method=prop_method, chkfile=chkfile)
        self.cell = mf.cell
        from pyscf.pbc.dft.multigrid import MultiGridNumInt
        if hasattr(mf, '_numint') and isinstance(mf._numint, MultiGridNumInt):
            raise NotImplementedError('Multigrid is not supported yet for RT-TDDFT')
        self.h1e_kin = None
        self.h1e_nuc_local = None
        self.h1e_ipovlp = None
        self.vgppnl_helper = None

    def init_onebody_integrals(self):
        """Cache one-body integrals: kinetic, nuclear (local part of pseudopotentials if applicable),
           and <nabla mu | nu>.
        """
        mf = self._scf
        cell = self.cell
        kpts = mf.kpts
        if cell.pseudo:
            self.h1e_nuc_local = get_pseudopotential_local_part(mf, kpts)
        else:
            self.h1e_nuc_local = mf.with_df.get_nuc(kpts)
        self.h1e_kin = np.asarray(cell.pbc_intor('int1e_kin', comp=1, hermi=1, kpts=kpts))
        self.h1e_ipovlp = np.asarray(cell.pbc_intor('int1e_ipovlp', comp=3, hermi=0, kpts=kpts))
        if cell.pseudo:
            self.vgppnl_helper = VelGaugePPNLHelper(cell, kpts)
            self.vgppnl_helper.build()


    def kernel(self, t_end, dt, t_start=0.0, efield=None, mo_basis=True, afield=None):

        self.init_onebody_integrals()

        bc = KBasisChanger(self._scf.get_ovlp(), self._scf.mo_coeff, to_orthonormal=True, nkpts=len(self._scf.kpts))
        log = logger.new_logger(self, self.verbose)

        # with self.mol.with_common_origin((0.0, 0.0, 0.0)):
        #     ao_dip = self.mol.intor_symmetric('int1e_r', comp=3)
        # mo_dip = bc.rotate_focklike(ao_dip)
        # charges = self.mol.atom_charges()
        # coords  = self.mol.atom_coords()
        # nucl_dip = np.einsum('i,ix->x', charges, coords)

        
        self.trace = {'t': [], 'dipole': [], 'dm': []}

        if t_end <= t_start:
            raise ValueError('t_end must be greater than t_start')
        
        nsteps = math.ceil((t_end - t_start) / dt)

        # chkf = h5py.File(self.chkfile, "w") if self.chkfile is not None else None
        # if chkf is not None:
        #     chkf.create_dataset('t', (0,), maxshape=(None,), dtype=np.float64, chunks=True)
        #     chkf.create_dataset('dipole', (0, 3), maxshape=(None, 3), dtype=np.complex128, chunks=True)
        #     chkf.create_dataset('dm', (0, self.mol.nao, self.mol.nao),
        #                         dtype=np.complex128,
        #                         maxshape=(None, self.mol.nao, self.mol.nao),
        #                         chunks=(1, self.mol.nao, self.mol.nao))

        def stepcallback(state):
            t = state.time
            dm = state.dm
            if mo_basis:
                dmao = bc.rev_denslike(dm)
            else:
                dmao = dm
            velocity = get_electronic_velocity(self.cell, afield(t), self._scf.kpts, self.h1e_ipovlp, dm=dmao, vgppnl_helper=self.vgppnl_helper)
            self.trace['t'].append(t)
            self.trace['dipole'].append(-velocity)
            self.trace['dm'].append(dm.copy())
            # if chkf is not None:
            #     chkf['t'].resize((chkf['t'].shape[0] + 1), axis=0)
            #     chkf['dipole'].resize((chkf['dipole'].shape[0] + 1), axis=0)
            #     chkf['dm'].resize((chkf['dm'].shape[0] + 1), axis=0)
            #     chkf['t'][-1] = t
            #     chkf['dipole'][-1] = np.asarray(dipole, dtype=np.complex128)
            #     chkf['dm'][-1] = np.asarray(dm, dtype=np.complex128)

        

        if self.prop is None:
            if self.prop_method in RTSCF_PROP_METHODS:
                self.prop = RTSCF_PROP_METHODS[self.prop_method]
            else:
                raise ValueError(f'prop_method {self.prop_method} not recognized')

        dm = self._scf.make_rdm1()
        h1e = self.h1e_nuc_local + self.h1e_kin
        S = self._scf.get_ovlp()
        get_veff = self._scf.get_veff

        if mo_basis:
            v_ext = make_vext_velgauge(self.cell, afield, self._scf.kpts, self.h1e_ipovlp, bc=bc, vgppnl_helper=self.vgppnl_helper)
            fock_init = bc.rotate_focklike(h1e + get_veff(dm=dm))
            dm = bc.rotate_denslike(dm)
            hkin_prop = bc.rotate_focklike(self.h1e_kin)
        else:
            v_ext = make_vext_velgauge(self.cell, afield, self._scf.kpts, self.h1e_ipovlp, vgppnl_helper=self.vgppnl_helper)
            fock_init = h1e + get_veff(dm=dm)
            hkin_prop = self.h1e_kin

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
                hkin = hkin_prop
            )
