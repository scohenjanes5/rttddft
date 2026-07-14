#!/usr/bin/env python

from pyscf import df
from pyscf.pbc import gto as pbcgto, dft as pbcdft
from pyscf.pbc.gto.pseudo.ppnl_velgauge import get_gth_pp_nl_velgauge
import numpy as np
import pytest
from rttddft.pbc.rttdbase import KRTTDSCF, kick_afield
from rttddft import rttdbase as rtb
from rttddft.propagators import magnus4 as m4


_KICK_FRAC = {
    'mmut': 0.0,
    'magnus2': 0.5,
    'magnus4': 0.5 - np.sqrt(3) / 6,
}


def _diamond_mf():
    cell = pbcgto.Cell()
    cell.atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
    cell.a = '''
    1.7850000000 1.7850000000 0.0000000000
    0.0000000000 1.7850000000 1.7850000000
    1.7850000000 0.0000000000 1.7850000000
    '''
    cell.pseudo = 'gth-hf-rev'
    cell.basis = {'C': [[0, (0.8, 1.0)],
                        [1, (1.0, 1.0)]]}
    cell.precision = 1e-10
    cell.build()
    kmesh = [2,1,1]
    kpts = cell.make_kpts(kmesh)
    mf = pbcdft.KRKS(cell, kpts=kpts, xc='pbe0').rs_density_fit(auxbasis=df.autoaux(cell)).run()
    return mf, kpts


@pytest.mark.parametrize("prop_method", ['magnus2', 'mmut', 'magnus4'])
def test_rttddft_diamond_ao_mo(prop_method):
    mf, kpts = _diamond_mf()

    step = 1.0
    afield = kick_afield(_KICK_FRAC[prop_method] * step, 0.0001, dir=(1.0,0.0,0.0))
    myrtd_ao = KRTTDSCF(mf, prop_method=prop_method)
    myrtd_ao.kernel(2.0, step, afield=afield, mo_basis=False)

    myrtd_mo = KRTTDSCF(mf, prop_method=prop_method)
    myrtd_mo.kernel(2.0, step, afield=afield, mo_basis=True)

    assert len(myrtd_ao.trace['t']) > 0
    assert len(myrtd_mo.trace['t']) == len(myrtd_ao.trace['t'])

    C = mf.mo_coeff
    for t1, dip1, t2, dip2 in zip(myrtd_mo.trace['t'], myrtd_mo.trace['dipole'],
                          myrtd_ao.trace['t'], myrtd_ao.trace['dipole']):
        assert abs(t1 - t2) < 1e-8
        assert np.allclose(dip1, dip2, atol=1e-6)

    for t1, dm_mo, t2, dm_ao in zip(myrtd_mo.trace['t'], myrtd_mo.trace['dm'],
                          myrtd_ao.trace['t'], myrtd_ao.trace['dm']):
        assert abs(t1 - t2) < 1e-8
        for k in range(len(kpts)):
            assert np.allclose(C[k] @ dm_mo[k] @ C[k].conj().T, dm_ao[k], atol=1e-6)


@pytest.mark.parametrize("mo_basis", [True, False])
def test_magnus4_commutator_includes_pp_nl(mo_basis):
    """Nonlocal GTH PP must enter the Magnus4 commutator via v_ext_nl."""
    mf, kpts = _diamond_mf()
    step = 1.0
    afield = kick_afield(_KICK_FRAC['magnus4'] * step, 0.0001, dir=(1.0,0.0,0.0))

    seen = {'v_ext_nl': None}

    def step_spy(state, h1e, v_ext, S, get_veff, dt, conv_tol=1e-5,
                 mo_basis=False, bc=None, logger=None, callback=None, **kwargs):
        seen['v_ext_nl'] = kwargs.get('v_ext_nl')
        return m4.step_magnus4(
            state, h1e, v_ext, S, get_veff, dt, conv_tol=conv_tol,
            mo_basis=mo_basis, bc=bc, logger=logger, callback=callback, **kwargs)

    rtb.RTSCF_PROP_METHODS['magnus4_spy'] = step_spy
    myrtd = KRTTDSCF(mf, prop_method='magnus4_spy')
    myrtd.kernel(2.0, step, afield=afield, mo_basis=mo_basis)

    v_nl = seen['v_ext_nl']
    assert v_nl is not None
    assert v_nl.shape[0] == len(kpts)
    assert np.linalg.norm(v_nl) > 0.0

    # Reference nonlocal PP at q=0 should match what the kernel passed
    from rttddft.lib import KBasisChanger
    myrtd.init_onebody_integrals()
    v_nl_ao = get_gth_pp_nl_velgauge(
        mf.cell, np.zeros(3), kpts=kpts, vgppnl_helper=myrtd.vgppnl_helper)
    if mo_basis:
        bc = KBasisChanger(mf.get_ovlp(), mf.mo_coeff, to_orthonormal=True, nkpts=len(kpts))
        v_nl_ref = bc.rotate_focklike(v_nl_ao)
    else:
        v_nl_ref = v_nl_ao
    assert np.allclose(v_nl, v_nl_ref, atol=1e-10)

    # Omitting V_nl from the commutator must change the trajectory
    myrtd_on = KRTTDSCF(mf, prop_method='magnus4')
    myrtd_on.kernel(2.0, step, afield=afield, mo_basis=mo_basis)

    def step_no_nl(state, h1e, v_ext, S, get_veff, dt, conv_tol=1e-5,
                   mo_basis=False, bc=None, logger=None, callback=None, **kwargs):
        kwargs = dict(kwargs)
        kwargs['v_ext_nl'] = None
        return m4.step_magnus4(
            state, h1e, v_ext, S, get_veff, dt, conv_tol=conv_tol,
            mo_basis=mo_basis, bc=bc, logger=logger, callback=callback, **kwargs)

    rtb.RTSCF_PROP_METHODS['magnus4_no_nl'] = step_no_nl
    myrtd_off = KRTTDSCF(mf, prop_method='magnus4_no_nl')
    myrtd_off.kernel(2.0, step, afield=afield, mo_basis=mo_basis)

    dip_on = np.stack(myrtd_on.trace['dipole'])
    dip_off = np.stack(myrtd_off.trace['dipole'])
    assert np.max(np.abs(dip_on - dip_off)) > 1e-10
