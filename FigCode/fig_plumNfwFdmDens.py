# pylint: disable=C,W
import DataObj as do
import astroUtils as au
import matplotlib.pyplot as plt
import numpy as np
import os
import tomllib

M_stars_str = '1e1'
a_stars_str = '3.85'
segue1 = False
N = 256

if segue1:
	simName = f"haloTest/haloTest_S1_{M_stars_str}_plummer_{N}"

else:
	simName = f"haloTest/Dyn_friction/5Kpc/haloTest_{M_stars_str}_plummer_a{a_stars_str}_{N}"

M_stars = float(M_stars_str) # mass of stars in solar masses
#a_stars = 0.1 # scale radius of stars in kpc
a_stars = float(a_stars_str) # scale radius of stars in kpc

if segue1:
	fraction_FDM = .5

	toml_dir = os.path.join(os.getcwd(), f"../../HaloConstructor_public/Data/Segue1_{N}_1e2m22_NFW")
	with open(os.path.join(toml_dir, "params.toml"), "rb") as f:
		data = tomllib.load(f)

	Rhalf = data["Rhalf"]
	Rs = float(data["Rs"].split('(')[1].split(')')[0])
	m22 = np.array([data["m22"]])
	rho0 = data["rho0"]
	L = float(data["L"].split('(')[1].split(')')[0])

	
else:
	# halo parameters, must match haloTest.py
	fraction_FDM = .5
	meandens = 2.775e+11*0.7**2 * 0.31 * 1e-9 # mean density
	Rhalf = 0.3 # half light radius
	Rs = 2. # scale radius in kpc
	Rvir = Rhalf/0.015
	Mvir = (4*np.pi/3)*200*meandens*Rvir**3 # virial mass in solar masses
	con = Rvir / Rs # concentration parameter
	rho0 = Mvir/4/np.pi/Rs**3 /(np.log(1+con)-con/(1+con)) # scale density in solar masses / kpc^3




def rho_Plummer(r):
	return 3*M_stars/(4*np.pi*a_stars**3) * (1 + r**2/a_stars**2)**(-5/2)

def M_enclosed_Plummer(r):
	return M_stars * r**3 / (r**2 + a_stars**2)**(3/2)


def rho_NFW(r):
	# the CDM component only sources (1 - fraction_FDM) of the external potential
	return au.NFW(r, Rs, rho0) * (1. - fraction_FDM)

def M_enclosed_NFW(r):
	return au.M_NFW(r, Rs, rho0) * (1. - fraction_FDM)


def Plot_Density_Profiles(d):
	"""
	Plot the density profiles of the field, the NFW halo and the Plummer
	profile for comparison.
	"""

	to_cpu = lambda x: np.asarray(x.get()) if hasattr(x, 'get') else np.asarray(x)

	r_grid   = to_cpu(d.R).ravel()
	rho_psi  = to_cpu(d.GetFieldDensity()).ravel()

	dx    = float(d.L) / int(d.N)
	r_max = float(d.L) / 2.          # last radius with full spherical coverage
	# inside a few cells the available radii are discrete (0.87dx, 1.66dx, ...),
	# so log bins there would be narrower than the gaps between them and come out
	# empty. Use cell-wide bins until the log bins are wider than a cell.
	r_lin = 4 * dx
	bins  = np.concatenate([np.arange(0., r_lin, dx),
							np.logspace(np.log10(r_lin), np.log10(r_max), 30)])

	which        = np.digitize(r_grid, bins)
	# bin centres are the mean radius of the cells that actually landed in the bin
	r_cent       = np.array([r_grid[which == i].mean() if np.any(which == i)
							else np.nan
							for i in range(1, len(bins))])
	rho_psi_prof = np.array([rho_psi[which == i].mean() if np.any(which == i)
							else np.nan
							for i in range(1, len(bins))])

	rho_plum = rho_Plummer(r_cent)
	rho_nfw  = to_cpu(rho_NFW(r_cent))





	return r_cent, rho_psi_prof, rho_plum, rho_nfw

def central_field_density(d, n_cells = 2.):
	"""
	Central field density, averaged over the cells within n_cells * dx of the
	box centre. Returned in M_sun / kpc^3.
	"""

	to_cpu = lambda x: np.asarray(x.get()) if hasattr(x, 'get') else np.asarray(x)

	r_grid  = to_cpu(d.R).ravel()
	rho_psi = to_cpu(d.GetFieldDensity()).ravel()
	dx      = float(d.L) / int(d.N)

	core = r_grid < n_cells * dx
	return rho_psi[core].mean(), r_grid, rho_psi, dx




if __name__ == "__main__":
	d = do.MeshDataObj(simName)
	initial_drop = 0
	d.LoadPsi(initial_drop)
	r_cent, rho_psi_prof, rho_plum, rho_nfw = Plot_Density_Profiles(d)
	rho_field_cent, r_grid, rho_psi, dx = central_field_density(d)


	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

	ax1.loglog(r_cent, rho_psi_prof, "o-", label=r"$|\psi|^2$ (field)")
	ax1.loglog(r_cent, rho_plum,      "--", label="Plummer (stars)")
	ax1.loglog(r_cent, rho_nfw,       ":", label="NFW (CDM)")
	ax1.axvline(a_stars, color="grey", ls=":", lw=1, label=r"$a_\star$")
	ax1.set_xlabel(r"$r$ [kpc]")
	ax1.set_ylabel(r"$\rho$ [$M_\odot\,\mathrm{kpc}^{-3}$]")

	ax1.axhline(rho_field_cent, color="blue", ls=":", lw=1, label=r"$\rho_\psi = {:.2e}$".format(rho_field_cent))
	ax1.axhline(rho_Plummer(0.), color="orange", ls="--", lw=1, label=r"$\rho_\star = {:.2e}$".format(rho_Plummer(0.)))

	ax1.set_title("Density Profiles at T = 0")

	final_drop = d.data_drops
	d.LoadPsi(final_drop)
	r_cent, rho_psi_prof, rho_plum, rho_nfw = Plot_Density_Profiles(d)
	rho_field_cent, r_grid, rho_psi, dx = central_field_density(d)
	
	ax2.loglog(r_cent, rho_psi_prof, "o-", label=r"$|\psi|^2$ (field)")
	ax2.loglog(r_cent, rho_plum,      "--", label="Plummer (stars)")
	ax2.loglog(r_cent, rho_nfw,       ":", label="NFW (CDM)")
	ax2.axvline(a_stars, color="grey", ls=":", lw=1, label=r"$a_\star$")
	ax2.set_xlabel(r"$r$ [kpc]")
	ax2.set_ylabel(r"$\rho$ [$M_\odot\,\mathrm{kpc}^{-3}$]")

	ax2.axhline(rho_field_cent, color="blue", ls=":", lw=1, label=r"$\rho_\psi = {:.2e}$".format(rho_field_cent))
	ax2.axhline(rho_Plummer(0.), color="orange", ls="--", lw=1, label=r"$\rho_\star = {:.2e}$".format(rho_Plummer(0.)))

	Tf = d.Tf

	ax2.set_title("Density Profiles at T = " + str(int(Tf)))

	ax1.legend()
	ax2.legend()

	plt.tight_layout()
	plt.savefig(f"/home/jdarne1/Eberhardt_Massive_Stars/SchroPo_public/Data/{simName}/density_profiles.png", dpi=150)
	plt.show()










