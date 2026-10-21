# pylint: disable=C,W,I
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import numpy as np 

Init_radius_str = '5Kpc'

a_str = round(float(Init_radius_str.replace('Kpc',''))/1.3, 2) # initial radius of the stars in kpc

simNames = [f"haloTest/Dyn_friction/{Init_radius_str}/haloTest_1e6_plummer_a{a_str}_256"]



colors = ['r','b','g','c','m','y','k']
labels = [r'$M_p = 10^1 M_\odot$', r'$M_p = 10^4 M_\odot$', r'$M_p = 10^5 M_\odot$', r'$M_p = 5\times10^5 M_\odot$', r'$M_p = 10^6 M_\odot$', r'$M_p = 10^7 M_\odot$']




def Density_Profile(d):
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

	return r_cent, rho_psi_prof



def Calculate_sigma(rho_field, r_cent):

    M_enc = np.trapezoid(4 * np.pi * r_cent**2 * rho_field, r_cent)

    sigma = np.sqrt(au.G * M_enc / r_cent)

    return sigma

			
if __name__ == "__main__":
	fo = pu.FigObj(3, 1, figsize = (12, 8))
	ax_dens, ax_gran, ax_sigma = None, None, None
	
	for i in range(len(simNames)):
		print(simNames[i])
		d = do.MeshDataObj(simNames[i])
		initial_drop = 0
		d.LoadPsi(initial_drop)
		
		r_cent, rho_field = Density_Profile(d)
		
		sigma = Calculate_sigma(rho_field, r_cent)

		m22 = 5
		m_axion = m22 * 1e-22 * au.eV2SolarMass
		
		gran_mass = (au.hbar / (m_axion * sigma))**3 * rho_field

		if i == 0:
			ax_dens, _ = fo.AddPlot(r_cent, rho_field, color = colors[i], alpha = 1)
			ax_gran, _ = fo.AddPlot(r_cent, gran_mass, color = colors[i], alpha = 1)
			ax_sigma, _ = fo.AddPlot(r_cent, sigma / au.kms2kpcMyr, color = colors[i], alpha = 1)	
		else:
			fo.AddLine(r_cent, rho_field, color = colors[i], label = labels[i], alpha = 1, ax=ax_dens)
			fo.AddLine(r_cent, gran_mass, color = colors[i], label = labels[i], alpha = 1, ax=ax_gran)
			fo.AddLine(r_cent, sigma / au.kms2kpcMyr, color = colors[i], label = labels[i], alpha = 1, ax=ax_sigma)

	init_radius = float(Init_radius_str.replace('Kpc',''))
	fo.AddVertLine(init_radius, label = r'$r_\mathrm{init} = ' + f"{init_radius}" + r'$', color = 'k', alpha = 0.5, ls = '--')

	fo.SetXLabel(r'$r$ [kpc]', ax=ax_dens)
	fo.SetYLabel(r'$\rho$ [$M_\odot \mathrm{kpc^{-3}}$]', ax=ax_dens)
	fo.SetLogLog(r_cent, rho_field)
	fo.legend(fontsize = 12, ax=ax_dens)
	fo.SetTitle(r'Density Profile of Halo and Plummer Sphere', ax=ax_dens)
	
	
	fo.SetXLabel(r'$r$ [kpc]', ax=ax_gran)
	fo.SetYLabel(r'$M_\mathrm{gran}$ [$M_\odot$]', ax=ax_gran)
	fo.SetLogLog(r_cent, gran_mass, ax=ax_gran)
	fo.legend(fontsize = 12, ax=ax_gran)
	fo.SetTitle(r'Granule Mass vs Radius for $m_a = 5 \times 10^{-22}$ eV', ax=ax_gran)

	fo.SetXLabel(r'$r$ [kpc]', ax=ax_sigma)
	fo.SetYLabel(r'$\sigma$ [kpc/Myr]', ax=ax_sigma)
	fo.SetLogLog(r_cent, sigma / au.kms2kpcMyr, ax=ax_sigma)
	fo.legend(fontsize = 12, ax=ax_sigma)
	fo.SetTitle(r'Velocity Dispersion vs Radius', ax=ax_sigma)



	fo.fig.suptitle(r'Granule Mass vs Radius for $m_a = 5 \times 10^{-22}$ eV')


	fo.show()
	fo.save(d.dataDir + '../granMass')


	



 