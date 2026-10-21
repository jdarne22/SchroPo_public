# pylint: disable=C,W
# this test is designed to check that our distribution
# is approximately stable under an external nfw potential

try:
	import cupy as np
except ImportError:
	import numpy as np
import tomllib

import numpy as np_
import astroUtils as au
import gridUtils as gu
import mathUtils as mu
import sysUtils as su
import os
import glob
import re
import sys
sys.path.insert(1, 'Solvers')
import Solvers.mesh_solver_vect as MS

# test is ready to run
# mestTest2 - control test
# meshTest2b - use new update rule

M_stars_str = '1e6'
a_stars_str = '0.005'
N = 256
n_stars_str = '1e4'

segue1 = True
Dyn_fric = False



if segue1:

	simName = f"starTest/Segue_1/starTest_S1_{M_stars_str}_plummer_astars_{a_stars_str}_{N}"

elif Dyn_fric:
	simName = f"starTest/Dyn_friction/5Kpc/starTest_{M_stars_str}_plummer_a{a_stars_str}_Np_{n_stars_str}_{N}"

else:
	simName = f"starTest/eri_II/starTest_{M_stars_str}_plummer_{N}_2"

M_stars = float(M_stars_str)
print(f"total mass of stars: {M_stars:.3e} Msun")
a_stars = float(a_stars_str) # scale radius of stars in kpc
n_stars = int(float(n_stars_str))


Checkpoint = True
checkpoint_dir = f'/home/jdarne1/Eberhardt_Massive_Stars/SchroPo_public/Data/{simName}'


data_drops = 30
padded = True
#CHANGED
cf = .1
nf = 1
C = au.G*4*np.pi
Tf = 2000.
gpu = True
fraction_FDM = .5


R_initial_star = 2.
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



def V_ext_func(R):
	return au.V_NFW(R, Rs, rho0)*(1. - fraction_FDM)

# function used to calculate cdm effect on stars
def M_encl_func(r):
	return au.M_NFW(r, Rs, rho0)*(1. - fraction_FDM)

def GetExternalPotential():
	R, _, _ = gu.sphrGrid(N, L, gpu = gpu)
	return V_ext_func(R)



def StarICs():
	cp = np_
	if gpu:
		cp = np	
	
	rng = cp.random.default_rng()
	M_Scale = rho0*L**3

	r = cp.zeros((n_stars, 3))
	v = cp.zeros((n_stars, 3))

	R_mag = cp.abs(cp.random.normal(0, R_initial_star, size = (n_stars)))
	r_hat = mu.random_unit_vectors(n_stars)
	r_hat = su.gpuThis(r_hat)
	v_vec = cp.random.normal(0, R_initial_star, size = (n_stars,3))
	V_mag = np.sqrt(M_Scale * au.G / R_mag) * 1e-6

	for i in range(n_stars):
		r[i,:] = R_mag[i]*r_hat[i]
		v[i,:] = V_mag[i]*v_vec[i]

	return r, v

def StarICs_Plummer(n_stars, a_stars):
	cp = np_
	if gpu:
		cp = np

	cp.random.seed(44)

	X1 = cp.random.uniform(0, 1, n_stars)
	X2 = cp.random.uniform(0, 1, n_stars)
	X3 = cp.random.uniform(0, 1, n_stars)

	a = a_stars * (X1**(-2/3) - 1)**(-0.5)

	z = (1 - 2*X2) * a
	x = cp.sqrt(a**2 - z**2) * cp.cos(2*cp.pi*X3)
	y = cp.sqrt(a**2 - z**2) * cp.sin(2*cp.pi*X3)

	v_esc = cp.sqrt(2*au.G*M_stars)*(a_stars**2+a**2)**(-1/4)

	v = []

	for star in range(n_stars):
		X4 = cp.random.uniform(0, 1)
		X5 = cp.random.uniform(0, 1)
		while X4**2 * (1 - X4**2)**3.5 <= 0.1*X5:
			X4 = cp.random.uniform(0, 1)
			X5 = cp.random.uniform(0, 1)
		
		q = X4

		v_mag = v_esc[star] * q

		X6 = cp.random.uniform(0, 1)
		X7 = cp.random.uniform(0, 1)

		v_z = (1 - 2*X6) * v_mag
		v_x = cp.sqrt(v_mag**2 - v_z**2) * cp.cos(2*cp.pi*X7)
		v_y = cp.sqrt(v_mag**2 - v_z**2) * cp.sin(2*cp.pi*X7)

		v.append([v_x, v_y, v_z])
	
	r = cp.array([x, y, z]).T
	r_mag = cp.sqrt(cp.sum(r**2, axis = 1))
	v = cp.array(v)
	v_mag = cp.sqrt(cp.sum(v**2, axis = 1))

	#Virial check
	#CHANGED

	m_star = M_stars / n_stars

	W = -cp.sum(m_star * r_mag * au.G*M_stars * r_mag / (r_mag**2 + a_stars**2)**(3/2))

	KE = 0.5 * cp.sum(m_star * v_mag**2)

	scale_factor = cp.sqrt(cp.abs(W)/(2*KE))

	v_mag *= scale_factor
	v *= scale_factor


	return r, v

def M_encl_psi_profile(psi, L, N):

    rho = np.sum(np.abs(psi)**2, axis=0) if psi.ndim == 4 else np.abs(psi)**2
    dV  = (L / N)**3
    R, _, _ = gu.sphrGrid(N, L, gpu=gpu)       # radius of each cell centre
    idx = np.argsort(R.ravel())
    r_sorted = R.ravel()[idx]
    m_cum    = np.cumsum(rho.ravel()[idx] * dV)
    return r_sorted, m_cum                    


def StarICs_Massless_Plummer(n_stars, a_stars, psi):

	cp = np_
	if gpu:
		cp = np

	cp.random.seed(42)

	X1 = cp.random.uniform(0, 1, n_stars)
	X2 = cp.random.uniform(0, 1, n_stars)
	X3 = cp.random.uniform(0, 1, n_stars)

	a = a_stars * (X1**(-2/3) - 1)**(-0.5)

	z = (1 - 2*X2) * a
	x = cp.sqrt(a**2 - z**2) * cp.cos(2*cp.pi*X3)
	y = cp.sqrt(a**2 - z**2) * cp.sin(2*cp.pi*X3)

	r = cp.array([x, y, z]).T

	r_sorted, M_encl_psi = M_encl_psi_profile(psi, L, N)

	M_encl = M_encl_func(a) + np.interp(a, r_sorted, M_encl_psi)

	vel_mag = np.sqrt(au.G * M_encl / a)

	X6 = cp.random.uniform(0, 1, n_stars)
	X7 = cp.random.uniform(0, 1, n_stars)

	v_z = (1 - 2*X6) * vel_mag
	v_x = cp.sqrt(vel_mag**2 - v_z**2) * cp.cos(2*cp.pi*X7)
	v_y = cp.sqrt(vel_mag**2 - v_z**2) * cp.sin(2*cp.pi*X7)

	v = cp.array([v_x, v_y, v_z]).T

	return r, v

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
		if os.path.isfile(v_path):
			return i, drops[i], v_path
	return None


def SetICs():
	cp = np_
	if gpu:
		cp = np
		
	#CHANGED
	mp = cp.ones(n_stars)*M_stars / n_stars
	#mp = cp.zeros(n_stars)

	psi = cp.zeros((nf,N,N,N)) + 0j

	if segue1:
		psi[0,:,:,:] = cp.load(f'Data/haloTest/Segue_1/haloTest_S1_{M_stars_str}_plummer_astars_{a_stars_str}_{N}/psi/drop5.npy')
	elif Dyn_fric:
		psi[0,:,:,:] = cp.load(f'Data/haloTest/Dyn_friction/5Kpc/haloTest_{M_stars_str}_plummer_a{a_stars_str}_{N}/psi/drop5.npy')
	else:
		psi[0,:,:,:] = cp.load(f'Data/haloTest/eri_II/haloTest_{M_stars_str}_plummer_{N}_2/psi/drop5.npy')

	s = MS.Solver()

	initial_drop = 0
	T_initial = 0.
	Tf_run = Tf
	drops_run = data_drops

	found = get_latest_drop(checkpoint_dir) if Checkpoint else None

	if found is not None:
		initial_drop, r_path, v_path = found
		if initial_drop >= data_drops:
			raise RuntimeError(f"checkpoint at drop {initial_drop}: sim already complete")
		r = cp.load(r_path)
		v = cp.load(v_path)
		T_initial = Tf * initial_drop / data_drops
		Tf_run = Tf - T_initial
		drops_run = data_drops - initial_drop
		print(f"resuming from drop {initial_drop} (T = {T_initial:.1f})")
	else:
		if Checkpoint:
			print("no checkpoint found, generating new ICs")
		r, v = StarICs_Plummer(n_stars, a_stars)

	s.SetParams(simName=simName, N=N, data_drops=drops_run, padded=padded,
		cf=cf, L=L, m22=m22, C=C, Tf=Tf_run, r=r, v=v,
		np=n_stars, mp=mp, gpu=gpu, a_stars=a_stars)

	s.D = 3
	s.M_encl_func = M_encl_func
	s.V_ext_mesh_func = V_ext_func
	s.M_encl_func_on = True
	s.explicit_particle_forces = True
	s.particle_particle_forces = True
	# CHANGED
	s.eps = Interpart_spacing(a_stars, M_stars, n_stars)
	print(f"interparticle spacing = {s.eps:.5f}")

	#s.eps = 0

	s.initial_drop = initial_drop
	s.T_initial = T_initial

	# psi is static under freezeDensity, so the original field is still valid on resume

	s.set_psi(psi)
	s.set_K()
	s.oldUpdateRule = True
	s.freezeDensity = True

	s.interp_order = 3    # 1 NGP, 2 CIC (default), 3 TSC, 4 PCS
	s.fd_order     = 6    # 2, 4 (default), 6, 8


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
	s.ShowDensity("frozen_final_density")
	