# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 

simNames = ["simAll/physTest1a_massless_256", "simAll/physTest1a_1e6_plummer_256_high_order", "simAll/physTest1a_2e6_plummer_256", "simAll/physTest1a_3e6_plummer_256", "simAll/physTest1a_1e7_plummer_256"]
colors = ['r','b','g','c','m','y','k']
M_stars = [0, 1e6, 2e6, 3e6, 1e7]


def PlotHalfLightRadius(d, fo, j, max_R, fo2, max_COM):
	data_drops = d.data_drops


	t = np.linspace(0, d.Tf, data_drops + 1)
	R_half = np.zeros(data_drops + 1)
	plum_COM = np.zeros(data_drops + 1)

	for i in range(data_drops + 1):
		r, v = d.LoadCorpData(i, center = False)

		plum_COM_xyz = np.sum(r, axis = 0) / r.shape[0]
		R = np.sqrt(np.sum(np.abs(r - plum_COM_xyz)**2, axis = 1))
		R_half[i] = np.median(R)
		plum_COM[i] = np.sqrt(np.sum(plum_COM_xyz**2))


	max_R = max(max_R, np.max(R_half))
	max_COM = max(max_COM, np.max(plum_COM))

	label = [r'$M = 0$', r'$M = 1\times10^6 M_\odot$', r'$M = 2\times10^6 M_\odot$', r'$M = 3\times10^6 M_\odot$', r'$M = 1\times10^7 M_\odot$'][j]
	if j == 0:
		fo.AddPlot(t, R_half, color = colors[j], label = label)
		fo2.AddPlot(t, plum_COM, color = colors[j], label = label)
	else:
		fo.AddLine(t, R_half, color = colors[j], label = label)
		fo2.AddLine(t, plum_COM, color = colors[j], label = label)

	return max_R, max_COM


def Get_central_field_density(d, n_cells = 2):

	initial_drop = 0

	d.LoadPsi(initial_drop)

	to_cpu = lambda x: np.asarray(x.get()) if hasattr(x, 'get') else np.asarray(x)

	rho = to_cpu(d.GetFieldDensity()).ravel()

	r_grid = to_cpu(d.R).ravel()

	dx = float(d.L) / d.N

	core = r_grid < n_cells * dx
	rho_field_cent = rho[core].mean()

	print(f"central field density: {rho_field_cent:.3e} Msun/kpc^3")


	return rho_field_cent


def Calculate_t_distrupt(M_stars, rho_field_cent, d, k, init_stellar_radius):

	if k == 0:

		initial_drop = 0

		r, v = d.LoadCorpData(initial_drop)

		init_stellar_radius = np.median(np.sqrt(np.sum(np.abs(r)**2, axis = 1)))

	print(f'Initial stellar median radius: {init_stellar_radius:.3e} kpc')

	m_axion = 5 * 10**-22 * au.eV2SolarMass

	sigma_estimate = (au.G * 4/3 * np.pi * rho_field_cent * (2 * np.pi * au.hbar)**2 / (m_axion)**2)**(1/4)

	print(f"estimated velocity dispersion: {sigma_estimate:.3e} kpc/Myr")

	lambda_db = 2 * np.pi * au.hbar / (m_axion * sigma_estimate)

	print(f"de Broglie wavelength: {lambda_db:.3e} kpc")

	T_disrupt = M_stars / (4 * np.pi / 3 * rho_field_cent * init_stellar_radius**3) * (lambda_db / sigma_estimate)

	print(f"disruption time: {T_disrupt:.3e} Myr")

	return T_disrupt, init_stellar_radius


if __name__ == "__main__":
	fo = pu.FigObj()
	fo2 = pu.FigObj()

	max_R = 0
	max_COM = 0
	T_disrupts = []
	init_stellar_radius = 0

	for i in range(len(simNames)):
		d = do.MeshDataObj(simNames[i])

		halo_test_sim = 'haloTest/haloTest_1e6_plummer_256_high_order'

		d_halo = do.MeshDataObj(halo_test_sim)

		rho_field_cent = Get_central_field_density(d_halo)

		T_disrupt, init_stellar_radius = Calculate_t_distrupt(M_stars[i], rho_field_cent, d, i, init_stellar_radius)
		T_disrupts.append(T_disrupt)

		max_R, max_COM = PlotHalfLightRadius(d, fo, i, max_R, fo2, max_COM)

	for i in range(len(simNames)):
		T_disrupt = T_disrupts[i]
		T_disrupt = int(float(T_disrupt))
		fo.AddVertLine(T_disrupt, ylims = [0, max_R * 1.1], color = colors[i], linestyle = '--')

	fo.SetYLabel(r'$R$')
	fo.SetXLabel(r'$t$')
	fo.SetYLim(yLow = 0, yHigh = max_R * 1.1)
	fo.legend(fontsize = 15)
	fo.show()

	fo.save(d.dataDir + f'halflightradius_compare')

	fo2.SetYLabel(r'$COM$')
	fo2.SetXLabel(r'$t$')
	fo2.SetYLim(yLow = 0, yHigh = max_COM * 1.4)
	fo2.legend(loc = 'upper left')
	fo2.show()

	fo2.save(d.dataDir + f'COM_compare')
