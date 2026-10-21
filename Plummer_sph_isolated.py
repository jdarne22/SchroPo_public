# pylint: disable=C,W
# this test is designed to check that our distribution is stable in
# isolation: zero external NFW potential, zero background FDM field.
# gravity for the stars comes only from their mutual N-body forces.


try:
	import cupy as np
except ImportError:
	import numpy as np
import numpy as np_
import astroUtils as au
import gridUtils as gu
import mathUtils as mu
import sysUtils as su
import os
import glob
import re
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, 'Solvers')
import Solvers.mesh_solver_vect as MS

# test is ready to run
# mestTest2 - control test
# meshTest2b - use new update rule
base_simName = "plummer_isolated_256_massive"
Checkpoint = True

N = 256
data_drops = 20
padded = True
#CHANGED
cf = .1
nf = 1
C = au.G*4*np.pi
Tf = 1000.
gpu = True
fraction_FDM = 0. # zero FDM background for this test
#CHANGED
n_stars = [int(1e4)]

# Plummer sphere for the sampled stars — matches haloTest.py
M_stars = 3e6   # total Plummer mass in solar masses
a_stars = 0.1   # Plummer scale radius in kpc

meandens = 2.775e+11*0.7**2 * 0.31 * 1e-9 # mean density
Rhalf = 0.3 # half light radius
# dervived parameters
### virial radius: using the relationship between the virial and half-light radius from https://arxiv.org/abs/1212.2980
Rs = 2. # scale radius in kpc
Rvir = Rhalf/0.015 
# need to adjust the mass so its easier to sim?
Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
con = Rvir / Rs # concentration parameter

cp = np_
if gpu:
	cp = np

def to_np(x):
	# matplotlib can't implicitly convert cupy device arrays; pull to host explicitly
	return x.get() if hasattr(x, 'get') else np_.asarray(x)

L = Rvir*2 / cp.sqrt(3)
R_initial_star = 2.
m22 = cp.array([5.0])
rho0 = Mvir/4/cp.pi/Rs**3 /(cp.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3


def StarICs_Plummer(n_stars, a_stars, M_stars):
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

	v_esc = cp.sqrt(2*au.G*M_stars)*(a**2+a_stars**2)**(-1/4)

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

def Plummer_sph_v_dist_analytic(a_stars, M_stars):

	r = cp.linspace(0, 5*a_stars, 1000)

	v_esc = lambda r: cp.sqrt(2*au.G*M_stars)*(r**2+a_stars**2)**(-1/4)

	vel_dist = lambda v, v_esc: 512/(7*np.pi) * v**2/v_esc**3 * (1 - v**2/v_esc**2)**(7/2)

	return vel_dist, v_esc

def Interpart_spacing(a_stars, M_stars, n_stars):
	rho_center = 3*M_stars/(4*np.pi*a_stars**3)
	central_spacing = (M_stars/n_stars/rho_center)**(1/3)
	return central_spacing


def get_latest_drop(checkpoint_dir):
	# psi is all-zero and frozen here, so only r and v need restoring

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


def SetICs(n_stars, a_stars, M_stars):
	cp = np_
	if gpu:
		cp = np

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
		r,v = StarICs_Plummer(n_stars, a_stars, M_stars)
	

	mp = cp.ones(n_stars)*M_stars / n_stars

	s = MS.Solver()
	### set simulation parameters
	s.SetParams(simName=simName, N = N, data_drops = drops_run, padded=padded,
	 cf=cf, L = L, m22 = m22, C = C, Tf = Tf_run, r = r, v = v,
	 np = n_stars, mp = mp, gpu = gpu, a_stars = a_stars)
	### set initial field
	s.D = 3
	# no s.M_encl_func / s.M_encl_func_on: zero external NFW potential
	s.explicit_particle_forces = True
	#CHANGED was 1e-5 now 1e-5/3 but now 1e-9
	s.eps = Interpart_spacing(a_stars, M_stars, n_stars)
	print(f"interparticle spacing = {s.eps:.5f}")
	s.initial_drop = initial_drop
	s.T_initial = T_initial

	s.interp_order = 3    # 1 NGP, 2 CIC (default), 3 TSC, 4 PCS
	s.fd_order     = 6    # 2, 4 (default), 6, 8



	# zero FDM field: psi stays all-zero and frozen for the whole run
	psi = cp.zeros((nf,N,N,N)) + 0j

	s.set_psi(psi)
	s.set_K()
	s.oldUpdateRule = True
	s.freezeDensity = True

	return s


if __name__ == "__main__":
	# set up sim ics

	for n_stars in n_stars:
		print(f"Running simulation with {n_stars} stars")

		# each run gets its own subfolder inside the base folder, with n_stars
		# appended to the name
		simName = f'Isolated_Plummer/{base_simName}/{base_simName}_{n_stars}'
		checkpoint_dir = f'/home/jdarne1/Eberhardt_Massive_Stars/SchroPo_public/Data/{simName}'
		os.makedirs(checkpoint_dir, exist_ok=True)

		s = SetICs(n_stars, a_stars, M_stars)

		vel_dist, v_esc = Plummer_sph_v_dist_analytic(a_stars, M_stars)

		v = s.v
		r = s.r

		v_mag = cp.sqrt(cp.sum(v**2, axis = 1))
		r_mag = cp.sqrt(cp.sum(r**2, axis = 1))

		a_tests = [0.3*a_stars, 0.5*a_stars, a_stars, 2*a_stars, 3*a_stars, 4*a_stars]

		masks = [cp.abs(r_mag - a) < 0.01*a for a in a_tests]

		fig, ax = plt.subplots(6, 1, figsize = (10, 13))

		for i in range(len(a_tests)):
			a = a_tests[i]
			mask = masks[i]
			masked_v_mag = v_mag[mask]
			v_grid = cp.linspace(0, v_esc(a), 1000)
			vel_dist_at_a = vel_dist(v_grid, v_esc(a))
			ax[i].plot(to_np(v_grid), to_np(vel_dist_at_a), label=f'Analytic Distribution', color='orange')
			ax[i].hist(to_np(masked_v_mag), bins=30, density=True, alpha=0.5, label=f'Sampled Velocities', color='blue')
			ax[i].axvline(to_np(v_esc(a)), color='r', linestyle='--', label=f'v_esc = {v_esc(a):.2f} kpc/Gyr')
			ax[i].set_xlabel('Velocity')
			ax[i].set_ylabel('Velocity Distribution')
			ax[i].legend()
			ax[i].set_title(f'Velocity Distribution at r = {a:.2f} kpc')




		# s.ShowDensity("initial_density")
		# s.Update(10.)
		# s.ShowDensity()
		# assert(0)

		# - run sim
		s.RunSim()
		#s.ShowDensity("frozen_final_density")
		
		plt.tight_layout()
		plt.savefig(f'{checkpoint_dir}/vel_dist_with_virial.png')
		plt.close()