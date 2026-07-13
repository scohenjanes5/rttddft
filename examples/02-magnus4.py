#!/usr/bin/env python

from pyscf import gto, scf, dft
from pyscf.data import nist
import numpy as np

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

from rttddft.rttdbase import RTTDSCF, gpulse_efield, kick_field
#efield = gpulse_efield(3, 0.001, 1, dir=(0,0,1.0), freq=0.0, phaseshift=0.0)

step = 0.4
# Place the kick on the first Gauss quadrature point so Magnus4 can sample it:
# t1 = (1/2 - √3/6) Δt
# NOTE: The intensity will be half of what other propagators would register due
# to the averaging over both t1 (has the kick) and t2 (no kick)
efield = kick_field((0.5 - np.sqrt(3)/6) * step, 0.0001, dir=(0,0,1.0))
myrtd = RTTDSCF(mf, chkfile='rtd.chk', prop_method='magnus4')

myrtd.kernel(100.0, step, efield=efield)

T = np.stack(myrtd.trace['t'])
dp = np.stack(myrtd.trace['dipole']).T
import matplotlib.pyplot as plt


for i in (2,):
    dpipad = np.pad(dp[i]-np.mean(dp[i]), (0, 50000 - dp[2].size), 'constant', constant_values=0)

    dpipad *= np.exp(-step/50.0 * np.arange(50000))

    rft = np.fft.rfft(dpipad)
    rftfreqs = np.fft.rfftfreq(50000, step)
    plt.plot(rftfreqs * nist.HARTREE2EV * 2 * np.pi, rft.imag)

plt.xlim(0, 27)
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (arb. units)')
plt.title(r'$E_z(\omega)$ for water molecule, 6-31G, PBE0')

plt.show()