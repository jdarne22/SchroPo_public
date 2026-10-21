# pylint: disable=C,W
# simulates stars + fdm + cdm
# represented as corpuscular particles, a field,
# and an external potential respectively


import tomllib

import numpy as np_
try:
	import cupy as np
except ImportError:
	import numpy as np
import astroUtils as au
import gridUtils as gu
import plotUtils as pu
import glob
import os
import re
import sys
sys.path.insert(1, 'Solvers')
import Solvers.mesh_solver_vect as MS

# sims
# - physTest1a: ics1a, fraction 0.5
# - physTest1b: ics1a, fraction 0.5
# - need to test physTest1a with old update

M_stars_str = '1e6'
a_stars_str = '0.005'
N = 256
n_stars_str = '1e4'

segue1 = True
dyn_fric = False

if segue1:
	simName = f"simAll/Segue_1/physTest1a_S1_{M_stars_str}_plummer_astars_{a_stars_str}_{N}"
elif dyn_fric:
	simName = f"simAll/Dyn_friction/5Kpc/physTest1a_{M_stars_str}_plummer_a{a_stars_str}_Np_{n_stars_str}_{N}"
else:
	simName = f"simAll/eri_II/physTest1a_{M_stars_str}_plummer_{N}_2"

Checkpoint = False
checkpoint_dir = f'/home/jdarne1/Eberhardt_Massive_Stars/SchroPo_public/Data/{simName}'

M_stars = float(M_stars_str)
print(f"total mass of stars: {M_stars:.3e} Msun")
a_stars = float(a_stars_str) # scale radius of stars in kpc
n_stars = int(float(n_stars_str))


data_drops = 50
padded = True
# CHANGED
cf = .1 
nf = 1
C = au.G*4*np.pi
Tf = 10000.
gpu = True
fraction_FDM = 0.5

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
	rho0_ = rho0
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
	rho0_ = Mvir/4/np_.pi/Rs**3 /(np_.log(1+con)-con/(1+con))



### EDIT ME
### IC info
# field_dir = 'Data/physTest1a/psi/drop10.npy'
# r_dir = 'Data/physTest1a/r/drop10.npy'
# v_dir = 'Data/physTest1a/v/drop10.npy'

if segue1:
	starTest_name = f'starTest' + simName.split('/')[-1].split('Test1a')[-1]
	r_dir              = f'Data/starTest/Segue_1/{starTest_name}/r/drop30.npy'
	v_dir              = f'Data/starTest/Segue_1/{starTest_name}/v/drop30.npy'
	field_dir          = f'Data/starTest/Segue_1/{starTest_name}/psi/drop30.npy'
elif dyn_fric:
	starTest_name = f"starTest_{M_stars_str}_plummer_a{a_stars_str}_Np_{n_stars_str}_{N}"
	r_dir              = f'Data/starTest/Dyn_friction/5Kpc/{starTest_name}/r/drop30.npy'
	v_dir              = f'Data/starTest/Dyn_friction/5Kpc/{starTest_name}/v/drop30.npy'
	field_dir          = f'Data/starTest/Dyn_friction/5Kpc/{starTest_name}/psi/drop30.npy'
else:
	starTest_name = f"starTest_{M_stars_str}_plummer_{N}_2"
	r_dir              = f'Data/starTest/eri_II/{starTest_name}/r/drop30.npy'
	v_dir              = f'Data/starTest/eri_II/{starTest_name}/v/drop30.npy'
	field_dir          = f'Data/starTest/eri_II/{starTest_name}/psi/drop30.npy'

initial_drop = 0
initial_time = 0.

#hbar_ = au.h_tilde(m22)[0]
#sigma = cp.sqrt(au.G * Mvir / Rvir)
#lam = hbar_ / sigma
#print(lam / sigma)

# function used to calculate cdm effect on fdm
def V_ext_func(R):
	return au.V_NFW(R, Rs, rho0)*(1. - fraction_FDM)

# function used to calculate cdm effect on stars
def M_encl_func(r):
	return au.M_NFW(r, Rs, rho0)*(1. - fraction_FDM)


def Interpart_spacing(a_stars, M_stars, n_stars):
	rho_center = 3*M_stars/(4*np.pi*a_stars**3)
	central_spacing = (M_stars/n_stars/rho_center)**(1/3)
	return central_spacing


def get_latest_drop(checkpoint_dir):

	drops = {}

	for path in glob.glob(os.path.join(checkpoint_dir, 'r', 'drop*.npy')):
		m = re.fullmatch(r'drop(\d+)\.npy', os.path.basename(path))
		if m:
			drops[int(m.group(1))] = path

	for i in sorted(drops, reverse=True):
		v_path = os.path.join(checkpoint_dir, 'v', f'drop{i}.npy')
		psi_path = os.path.join(checkpoint_dir, 'psi', f'drop{i}.npy')
		if os.path.isfile(v_path) and os.path.isfile(psi_path):
			return i, drops[i], v_path, psi_path

	return None


def SetICs():

	cp = np_
	if gpu:
		cp = np

	drop = initial_drop
	T_initial = initial_time
	Tf_run = Tf
	drops_run = data_drops

	found = get_latest_drop(checkpoint_dir) if Checkpoint else None

	if found is not None:
		drop, r_src, v_src, field_src = found
		if drop >= data_drops:
			raise RuntimeError(f"checkpoint at drop {drop}: sim already complete")
		T_initial = Tf * drop / data_drops
		Tf_run = Tf - T_initial
		drops_run = data_drops - drop
		print(f"resuming from drop {drop} (T = {T_initial:.1f})")
	else:
		if Checkpoint:
			print("no checkpoint found, starting from frozenHalo output")
		r_src, v_src, field_src = r_dir, v_dir, field_dir

	psi = cp.zeros((nf,N,N,N)) + 0j
	r = cp.load(r_src)
	r %= L
	r[r < -1*L/2.] += L
	r[r > L/2.] -= L
	v = cp.load(v_src)
	n_p = len(v)
	psi[0,:,:,:] = cp.load(field_src)

	s = MS.Solver()
	### set simulation parameters
	s.SetParams(simName=simName, N = N, data_drops = drops_run, padded=padded,
	 cf=cf, L = L, m22 = m22, C = C, Tf = Tf_run, gpu = gpu, r = r, v = v, np = n_p,
	 mp = cp.ones(n_p)*M_stars / n_p, a_stars = a_stars)
	### set initial field
	s.D = 3
	s.M_encl_func = M_encl_func # used for particle forces
	s.V_ext_mesh_func = V_ext_func # used for field forces
	s.M_encl_func_on = True
	s.explicit_particle_forces = True
	s.particle_particle_forces = True

	# CHANGED
	s.eps = Interpart_spacing(a_stars, M_stars, n_p)
	print(f"interparticle spacing = {s.eps:.5f}")
	
	#s.eps = 0

	s.initial_drop = drop
	s.T_initial = T_initial

	s.set_psi(psi)
	s.set_K()

	s.interp_order = 3    # 1 NGP, 2 CIC (default), 3 TSC, 4 PCS
	s.fd_order     = 6    # 2, 4 (default), 6, 8


	extras = {}
	extras['fraction_FDM'] = fraction_FDM
	extras['r_ic_directory'] = r_src
	extras['v_ic_directory'] = v_src
	extras['field_ic_directory'] = field_src
	extras['R'] = a_stars
	extras['M'] = M_stars
	extras['Rs'] = Rs
	extras['rho0'] = rho0_
	s.extras = extras
	
	s.oldUpdateRule = True

	s.R,_,_ = gu.sphrGrid(N,L,gpu = gpu)

	return s


if __name__ == "__main__":
	# set up sim ics
	s = SetICs()
	# - run sim
	s.RunSim()