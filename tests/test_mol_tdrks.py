from pyscf import gto, dft
import numpy as np
import pytest
from rttddft.rttdbase import RTTDSCF, kick_field


_KICK_FRAC = {
    'mmut': 0.0,
    'magnus2': 0.5,
    'magnus4': 0.5 - np.sqrt(3) / 6,
}


@pytest.mark.parametrize("prop_method", ['magnus2', 'mmut', 'magnus4'])
def test_rttddft_water(prop_method):

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
    efield = kick_field(_KICK_FRAC[prop_method] * step, 0.0001, dir=(0,0,1.0))
    myrtd = RTTDSCF(mf, prop_method=prop_method)

    myrtd.kernel(4.0, step, efield=efield)
    assert len(myrtd.trace['t']) > 0
    assert len(myrtd.trace['dipole']) == len(myrtd.trace['t'])
