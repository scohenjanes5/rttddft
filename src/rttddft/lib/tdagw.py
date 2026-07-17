import math
import numpy as np
import scipy.linalg as sla
import scipy
import h5py
from rttddft.rttdbase import RTTDSCF, get_mo_dip, make_bc, make_vext_from_efield, RTSCF_PROP_METHODS

from rttddft.propagators import magnus2
from rttddft.lib import BasisChanger

import pyscf.lib
from pyscf.data import nist
from pyscf.lib import logger

einsum = pyscf.lib.einsum


def vhartree_and_screened_exchange(G, Lpq, Lpqbar):
    naux, _, _ = Lpq.shape
    nmo = Lpq.shape[1]

    # contraction: V
        # The following code is exactly equivalent to
        # Lpq_z[s] = einsum('Pjb,jb->P', Lia[s], z)
    #Lpq_z[s] = Lia[s].reshape(naux, -1) @ z.reshape(-1)
    # Lpq_z = Lpq.reshape(naux, -1) @ G.real.reshape(-1) + \
    #         1j * (Lpqbar.reshape(naux, -1) @ G.imag.reshape(-1))

    # vz = Lpq.reshape(naux, -1).T @ Lpq_z.real + \
    #     1j * (Lpqbar.reshape(naux, -1).T @ Lpq_z.imag)
    # vz = vz.reshape(nmo, nmo)

    Gr = np.ascontiguousarray(G.real)
    Gi = np.ascontiguousarray(G.imag)


    zr = Lpq.reshape(naux, -1) @ Gr.reshape(-1)
    vzr = Lpq.reshape(naux, -1).T @ zr
    zi = Lpq.reshape(naux, -1) @ Gi.reshape(-1)
    vzi = Lpq.reshape(naux, -1).T @ zi

    vz = vzr + 1j * vzi
    vz = vz.reshape(nmo, nmo)

    # contraction: W

    # continue
    # this matrix can be large
    # Lpq_z = np.zeros(shape=[nspin, naux, nmo, nmo], dtype=np.double)


    # The following calculation for waz is equivalent to
    # Laj_zs = einsum('Lab,jb->Laj', Laa[s], z)
    # waz = -einsum('Lji,Laj->ia', Lii_bar[s], Laj_zs).reshape(-1)

    # the following calculation for wbz is equivalent to
    # Lij_zs = einsum('Lib,jb->Lij', Lia[s], z)
    # wbz = -einsum('Lja,Lij->ia', Lia_bar[s], Lij_zs).reshape(-1)


    # Laj_zs = einsum('Lab,jb->Laj', Lpq, G)
    # waz = -einsum('Lji,Laj->ia', Lpqbar, Laj_zs).reshape(nmo, nmo)



    #waz = -einsum('Lnr, Lms, rs->nm', Lpqbar, Lpq, G)
    waz = -einsum('Lnr, Lms, rs->nm', Lpqbar, Lpq, Gr) + \
        -1j * einsum('Lnr, Lms, rs->nm', Lpqbar, Lpq, Gi)


    return vz, waz



def get_lpq_bar(nocc, mo_energy, Lpq):
    """Calculate the auxiliary 3-center matrix.
    Lpq_bar = (epsilon)^-1 * Lpq
    Equation 11 in doi.org/10.1002/jcc.24688.

    Args:
        nocc (int array): the number of occupied orbitals.
        mo_energy (double ndarray): orbital energy.
        Lpq (double ndarray): 3-center density-fitting matrix.

    Returns:
        Lpq_bar (double ndarray): auxiliary three-center matrix.
    """
    naux, _, _ = Lpq.shape

    # calculate the response function in the auxiliary basis
    orb_diff = mo_energy[: nocc, None] - mo_energy[None, nocc :]
    orb_diff = 1.0 / orb_diff

    X = 4.0 * einsum('Pia,ia,Qia->PQ', Lpq[:, : nocc, nocc :], orb_diff, Lpq[:, : nocc, nocc :])


    # calculate the inverse dielectric function
    InvD = np.linalg.inv((np.eye(naux) - X))

    # calculate the auxiliary matrix
    Lpq_bar = einsum('PQ,Qmn->Pmn', InvD, Lpq)

    return Lpq_bar



class TDAGW(RTTDSCF):



    def __init__(self, gw, prop_method='magnus2', **kwargs):
        super().__init__(gw._scf, prop_method, **kwargs)
        self.gw = gw

    def kernel(self, t_end, dt, t_start=0.0, efield=None, **kwargs):
        bc = make_bc(self._scf)
        mo_dip = get_mo_dip(bc, self.mol)
        v_ext = make_vext_from_efield(efield, mo_dip)
        log = logger.new_logger(self, self.verbose)


        self.trace = {'t': [], 'dipole': [], 'dm': []}

        if t_end <= t_start:
            raise ValueError('t_end must be greater than t_start')
        
        nsteps = math.ceil((t_end - t_start) / dt)

        chkf = h5py.File(self.chkfile, "w") if self.chkfile is not None else None
        if chkf is not None:
            chkf.create_dataset('t', (0,), maxshape=(None,), dtype=np.float64, chunks=True)
            chkf.create_dataset('dipole', (0, 3), maxshape=(None, 3), dtype=np.complex128, chunks=True)
            chkf.create_dataset('dm', (0, self.mol.nao, self.mol.nao),
                                dtype=np.complex128,
                                maxshape=(None, self.mol.nao, self.mol.nao),
                                chunks=(1, self.mol.nao, self.mol.nao))
        def stepcallback(t, dm):
            dipole = -np.sum(mo_dip * dm[None,:,:].real, axis=(1,2))
            self.trace['t'].append(t)
            self.trace['dipole'].append(dipole)
            self.trace['dm'].append(dm.copy())
            if chkf is not None:
                chkf['t'].resize((chkf['t'].shape[0] + 1), axis=0)
                chkf['dipole'].resize((chkf['dipole'].shape[0] + 1), axis=0)
                chkf['dm'].resize((chkf['dm'].shape[0] + 1), axis=0)
                chkf['t'][-1] = t
                chkf['dipole'][-1] = np.asarray(dipole, dtype=np.complex128)
                chkf['dm'][-1] = np.asarray(dm, dtype=np.complex128)

        #hcore = self.mol.intor('int1e_kin') + self.mol.intor('int1e_nuc')
        #hcore_mo = bc.rotate_focklike(hcore)
        #scissor = np.diag(self.gw.mo_energy - self._scf.mo_energy)
        hcore_mo = np.diag(self.gw.mo_energy)

        G0 = bc.rotate_denslike(self._scf.make_rdm1())

        Lpq = self.gw.Lpq
        Lpqbar = get_lpq_bar(self.mol.nelectron//2, self.gw.mo_energy, Lpq)
        vz0, waz0 = vhartree_and_screened_exchange(G0, Lpq, Lpqbar)

        def get_veff(dm=None):
            vz, waz = vhartree_and_screened_exchange(dm, Lpq, Lpqbar)
            return (vz-vz0) + 0.5*(waz-waz0)

        if self.prop is None:
            if self.prop_method in RTSCF_PROP_METHODS:
                self.prop = RTSCF_PROP_METHODS[self.prop_method](
                    h1e = hcore_mo,
                    get_veff = get_veff,
                    dm = G0,
                    bc = bc,
                    all_mo_basis=True,
                    v_ext = v_ext,
                    dt = dt,
                    stepcallback=stepcallback,
                    logger = log
                )
            else:
                raise ValueError(f'prop_method {self.prop_method} not recognized')

        if t_end <= t_start:
            raise ValueError('t_end must be greater than t_start')

        nsteps = math.ceil((t_end - t_start) / dt)

        for _ in range(nsteps):
            self.prop.step()


if __name__ == '__main__':
    from pyscf import gto, dft
    from fcdmft.gw.mol.gw_cd import GWCD


    mol = gto.Mole()
    mol.build(
        atom = '''
        O     0.00000000    -0.00001441    -0.34824012
        H    -0.00000000     0.76001092    -0.93285191
        H     0.00000000    -0.75999650    -0.93290797
        ''',
        basis = 'cc-pvdz',
        verbose=5,
    )

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    # diag self-energy, incore
    gw = GWCD(mf)
    gw.vhf_df = True
    gw.kernel()




    step = 0.4
    estrength = 0.00001
    from rttddft.rttdbase import kick_field
    efield = kick_field(0, estrength, dir=(0,0,1.0))


    tdgw = TDAGW(gw, prop_method='mmut')
    tdgw.chkfile = 'tdgw.chk'

    tdgw.kernel(400, step, efield=efield)
    T = np.stack(tdgw.trace['t'])
    dms = np.stack(tdgw.trace['dm'])
    dp = np.stack(tdgw.trace['dipole']).T
    # import h5py
    # with h5py.File("/home/chillenb/src/rttddft/data/tdgw.h5", "w") as f:
    #     f.create_dataset("t", data=T)
    #     f.create_dataset("dipole", data=dp)
    #     f.create_dataset("dm", data=dms)

    # tddft = RTTDSCF(mf)

    # tddft.kernel(100.0, step, efield=efield)
    # T2 = np.stack(tddft.trace['t'])
    # dp2 = np.stack(tddft.trace['dipole']).T
    import matplotlib.pyplot as plt

    from fcdmft.gw.mol.bse import BSE
    mybse = BSE(gw)
    mybse.residue_thresh = 1e-10
    exci, _, _ = mybse.kernel('s')
    dip, osc = mybse.get_oscillator_strength()
    print(dip[2])
    print(exci)


    for i in (2,):
        #dpipad = np.pad(dp[i]-np.mean(dp[i]), (0, 100000 - dp[2].size), 'constant', constant_values=0)
        dpipad = dp[i]-np.mean(dp[i])
        #dpipad2 = np.pad(dp2[i]-np.mean(dp2[i]), (0, 50000 - dp[2].size), 'constant', constant_values=0)

        # dpipad *= np.exp(-0.05/50.0 * np.arange(100000))
        #dpipad2 *= np.exp(-0.2/50.0 * np.arange(50000))


        #rft2 = np.fft.rfft(dpipad2)
        tau = -(dpipad.size-1) / np.log(0.001)
        window = scipy.signal.windows.exponential(dpipad.size, center=0, tau = tau, sym=False)
        dpipad = np.pad(dpipad*window, (0, 100000 - dpipad.size), 'constant', constant_values=0)

        rft = np.fft.fft(dpipad)
        rftfreqs = np.fft.fftfreq(dpipad.size, step)

        import IPython
        IPython.embed()
        maxval = np.max(rft.imag)
        plt.plot(rftfreqs * nist.HARTREE2EV * 2 * np.pi, rft.imag / estrength)
        plt.stem(exci* nist.HARTREE2EV, np.abs(dip[2]), markerfmt='ro', basefmt='r-', linefmt='r-')

    plt.xlim(0, 27)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity (arb. units)')
    plt.title(r'$E_z(\omega)$ for water molecule, 6-31G, TD-aGW/G0W0@PBE0')

    plt.show()