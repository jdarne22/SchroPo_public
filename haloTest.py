# pylint: disable=C,W
# this test is designed to check that our distribution
# is approximately stable under an external nfw potential


import tomllib

import numpy as np_
try:
	import cupy as np
except ImportError:
	import numpy as np
import astroUtils as au
import gridUtils as gu
import sys
sys.path.insert(1, 'Solvers')
import Solvers.mesh_solver_vect as MS
import os
import glob
import re

# test is ready to run
# mestTest2 - control test
# meshTest2b - use new update rule

M_stars_str = '1e6'
a_stars_str = '0.005'
N = 256


segue1 = True
Dyn_fric = False

if segue1:

	simName = f"haloTest/Segue_1/haloTest_S1_{M_stars_str}_plummer_astars_{a_stars_str}_{N}"

elif Dyn_fric:
	simName = f"haloTest/Dyn_friction/5Kpc/haloTest_{M_stars_str}_plummer_a{a_stars_str}_{N}"

else:
	simName = f"haloTest/eri_II/haloTest_{M_stars_str}_plummer_{N}_2"

M_stars = float(M_stars_str)
print(f"total mass of stars: {M_stars:.3e} Msun")
a_stars = float(a_stars_str) # scale radius of stars in kpc


Checkpoint = True
checkpoint_dir = f'/home/jdarne1/Eberhardt_Massive_Stars/SchroPo_public/Data/{simName}'
data_drops = 5
padded = True
#CHANGED
cf = .1
nf = 1
C = au.G*4*np.pi
Tf = 1500.
gpu = True

fraction_FDM = .5


meandens = 2.775e+11*0.7**2 * 0.31 * 1e-9 # mean density

cp = np_
if gpu:
	cp = np

if segue1:
	toml_dir = os.path.join(os.getcwd(), f"../HaloConstructor_public/Data/Segue1_{N}_1e2m22_NFW")
	with open(os.path.join(toml_dir, "params.toml"), "rb") as f:
		data = tomllib.load(f)

	Rhalf = data["Rhalf"]
	Rs = float(data["Rs"].split('(')[1].split(')')[0])
	m22 = cp.array([data["m22"]])
	rho0 = data["rho0"]
	L = float(data["L"].split('(')[1].split(')')[0])
	
else:
	Rhalf = 0.3 # half light radius
	Rs = 2. # scale radius in kpc
	m22 = cp.array([5.0])

	Rvir = Rhalf/0.015 
	# need to adjust the mass so its easier to sim?
	Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
	print(f"virial mass of halo: {Mvir:.3e} Msun within {Rvir:.3e} kpc")
	con = Rvir / Rs # concentration parameter
	L = Rvir*2 / cp.sqrt(3) # box size in kpc

	rho0 = Mvir/4/cp.pi/Rs**3 /(cp.log(1+con)-con/(1+con)) 


def V_Plummer(R):
	return -au.G*M_stars / cp.sqrt(R**2 + a_stars**2)

def rho_Plummer(r):
	return 3*M_stars/(4*cp.pi*a_stars**3) * (1 + r**2/a_stars**2)**(-5/2)

def M_Plummer(r):
	return M_stars * r**3 / (r**2 + a_stars**2)**(3/2)


def V_ext_func(R):
	return au.V_NFW(R, Rs, rho0)*(1. - fraction_FDM) + V_Plummer(R)

# function used to calculate cdm effect on stars
def M_encl_func(r):
	return au.M_NFW(r, Rs, rho0)*(1. - fraction_FDM) + M_Plummer(r)

# def V_ext_func(R):
# 	return au.V_NFW(R, Rs, rho0)*(1. - fraction_FDM)

# def M_encl_func(r):
# 	return au.M_NFW(r, Rs, rho0)*(1. - fraction_FDM)

#central_dens_plummer = rho_Plummer(0.)
#print(f"central density of plummer profile: {central_dens_plummer:.3e} Msun/kpc^3")

#print(f"Mass of plummer sphere: {M_Plummer(20.):.3e} Msun within 20 kpc")

#print(f"virial mass of halo: {Mvir:.3e} Msun within {Rvir:.3e} kpc")


def get_latest_drop(checkpoint_dir):

	drops = {}

	for path in glob.glob(os.path.join(checkpoint_dir, 'psi', 'drop*.npy')):
		m = re.fullmatch(r'drop(\d+)\.npy', os.path.basename(path))
		if m:
			drops[int(m.group(1))] = path

	for i in sorted(drops, reverse=True):
		return i, drops[i]
	return None


def SetICs():
	cp = np_
	if gpu:
		cp = np
	s = MS.Solver()

	initial_drop = 0
	T_initial = 0.
	Tf_run = Tf
	drops_run = data_drops

	found = get_latest_drop(checkpoint_dir) if Checkpoint else None

	if found is not None:
		initial_drop, psi_path = found
		if initial_drop >= data_drops:
			raise RuntimeError(f"checkpoint at drop {initial_drop}: sim already complete")
		psi = cp.load(psi_path)
		T_initial = Tf * initial_drop / data_drops
		Tf_run = Tf - T_initial
		drops_run = data_drops - initial_drop
		print(f"resuming from drop {initial_drop} (T = {T_initial:.1f})")
	else:
		if Checkpoint:
			print("no checkpoint found, generating new ICs")
		psi = cp.zeros((nf,N,N,N)) + 0j

		if segue1:
			IC_dir = toml_dir
			psi[0,:,:,:] = cp.load(os.path.join(IC_dir, f'psi.npy')) * cp.sqrt(fraction_FDM)

		else:
			IC_dir = os.path.join(os.getcwd(), "../HaloConstructor_public/Code")
			psi[0,:,:,:] = cp.load(os.path.join(IC_dir, f'eri_200Emax_{N}_2.npy')) * cp.sqrt(fraction_FDM)


	mass_tot = cp.sum(cp.abs(psi)**2) * (L/N)**3
	print(f"total mass of FDM: {mass_tot:.3e} Msun")

	### set simulation parameters
	s.SetParams(simName=simName, N = N, data_drops = drops_run, padded=padded,
	 cf=cf, L = L, m22 = m22, C = C, Tf = Tf_run, gpu = gpu, a_stars = a_stars)
	### set initial field
	s.D = 3
	s.M_encl_func = M_encl_func
	s.V_ext_mesh_func = V_ext_func
	s.M_encl_func_on = True
	s.initial_drop = initial_drop
	s.T_initial = T_initial

	s.interp_order = 3    # 1 NGP, 2 CIC (default), 3 TSC, 4 PCS
	s.fd_order     = 6    # 2, 4 (default), 6, 8


	s.set_psi(psi)

	field_dens = s.GetFieldDensity()
	central_soliton_dens = cp.max(field_dens)
	print(f"central soliton density: {central_soliton_dens:.3e} Msun/kpc^3")

	s.set_psi(psi)
	s.set_K()
	s.oldUpdateRule = True

	return s


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()

	# s.ShowDensity("initial_density")
	# s.Update(10.)
	# s.ShowDensity()
	# assert(0)

	# - run sim
	s.RunSim()
	s.ShowDensity("final_density")

	