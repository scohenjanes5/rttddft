from pyscf import gto, dft
import numpy as np
import pytest
from rttddft.rttdbase import RTTDSCF, kick_field


@pytest.mark.parametrize("prop_method", ['magnus2', 'mmut', 'magnus4'])
def test_rttddft_water_ao_mo(prop_method):

    mol = gto.Mole()
    mol.build(
        atom = '''
        O     0.00000000    -0.00001441    -0.34824012
        H    -0.00000000     0.76001092    -0.93285191
        H     0.00000000    -0.75999650    -0.93290797
        ''',
        basis = '6-31G',
        symmetry = True,
        verbose=5,
    )

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    step = 0.4
    efield = kick_field(0.0, 0.0001, dir=(0,0,1.0))

    myrtd_ao = RTTDSCF(mf, prop_method=prop_method)
    myrtd_ao.kernel(4.0, step, efield=efield, mo_basis=False)

    myrtd_mo = RTTDSCF(mf, prop_method=prop_method)
    myrtd_mo.kernel(4.0, step, efield=efield, mo_basis=True)

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
        assert np.allclose(C @ dm_mo @ C.conj().T, dm_ao, atol=1e-6)
