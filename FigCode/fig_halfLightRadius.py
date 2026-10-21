# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np 
from scipy.optimize import brentq

# simNames = ["simAll/Dyn_friction/5Kpc/physTest1a_1e1_plummer_a3.85_Np_1e2_256", "simAll/Dyn_friction/5Kpc/physTest1a_1e2_plummer_a3.85_Np_1e2_256", "simAll/Dyn_friction/5Kpc/physTest1a_1e3_plummer_a3.85_Np_1e2_256",
# 			"simAll/Dyn_friction/5Kpc/physTest1a_1e4_plummer_a3.85_Np_1e2_256", "simAll/Dyn_friction/5Kpc/physTest1a_1e5_plummer_a3.85_Np_1e2_256", "simAll/Dyn_friction/5Kpc/physTest1a_1e6_plummer_a3.85_Np_1e2_256", 
# 			"simAll/Dyn_friction/5Kpc/physTest1a_1e7_plummer_a3.85_Np_1e2_256", "simAll/Dyn_friction/5Kpc/physTest1a_5e7_plummer_a3.85_Np_1e2_256", "simAll/Dyn_friction/5Kpc/physTest1a_1e9_plummer_a3.85_Np_1e2_256"]
simNames = ["simAll/eri_II/physTest1a_massless_256", "simAll/eri_II/physTest1a_1e6_plummer_256", "simAll/eri_II/physTest1a_2e6_plummer_256_2", "simAll/eri_II/physTest1a_3e6_plummer_256", "simAll/eri_II/physTest1a_1e7_plummer_256"]
simNames = ["simAllTest/Segue_1/physTest1a_S1_1e6_plummer_astars_0.005_256"]

colors = ['r','b','g','c','m','y','k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
#labels = [r'Mp = $10^{-1} M_\odot$', r'Mp = $1 M_\odot$', r'Mp = $10 M_\odot$', r'Mp = $10^2 M_\odot$',r'Mp = $10^3 M_\odot$', r'Mp = $10^4 M_\odot$', r'Mp = $10^5 M_\odot$', r'Mp = $5\times10^5 M_\odot$', r'Mp = $10^7 M_\odot$']
labels = [r'M = $0 M_\odot$', r'M = $1\times10^{6} M_\odot$', r'M = $2\times10^{6} M_\odot$', r'M = $3\times10^{6} M_\odot$', r'M = $10^{7} M_\odot$']
max_R = 0

Show_T_Equality = False



def PlotHalfLightRadius(d, fo, j, fo2, max_R, Show_T_Equality):
	data_drops = d.data_drops


	t = np.linspace(0, d.Tf, data_drops + 1)
	R_half = np.zeros(data_drops + 1)
	plum_COM = np.zeros(data_drops + 1)
	R_parts = np.zeros((data_drops + 1, d.np))
	R_half_non_cent = np.zeros(data_drops + 1)
	R_half_projected = np.zeros(data_drops + 1)

	x = d.dx * (0.5 + np.arange(-d.N//2, d.N//2))   # cell centres, same as meshSolver
	X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
	to_cpu = lambda x: np.asarray(x.get()) if hasattr(x, 'get') else np.asarray(x)

	dm_enclosed_mass = np.zeros(data_drops + 1)

	stellar_enclosed_mass = np.zeros(data_drops + 1)


	for i in range(data_drops + 1):
		r, v = d.LoadCorpData(i, center = False)

		plum_COM_xyz = np.sum(r, axis = 0) / r.shape[0]
		R = np.sqrt(np.sum(np.abs(r - plum_COM_xyz)**2, axis = 1))
		R_non_cent = np.sqrt(np.sum(np.abs(r)**2, axis = 1))
		R_half_non_cent[i] = np.median(R_non_cent)
		R_half[i] = np.median(R)
		R_parts[i] = R_non_cent
		plum_COM[i] = np.sqrt(np.sum(plum_COM_xyz**2))
		max_R = max(max_R, np.max(R_half))

		R_half_projected[i] = np.median(R * 3/4)

		if Show_T_Equality:

			# DM mass within R_half of the stellar COM
			d.LoadPsi(i)
			rho_psi = to_cpu(d.GetFieldDensity())          # shape (N, N, N), not raveled

			# periodic (minimum-image) offsets from the stellar COM
			dX = (X - plum_COM_xyz[0] + d.L/2) % d.L - d.L/2
			dY = (Y - plum_COM_xyz[1] + d.L/2) % d.L - d.L/2
			dZ = (Z - plum_COM_xyz[2] + d.L/2) % d.L - d.L/2
			r_grid = np.sqrt(dX**2 + dY**2 + dZ**2)

			dm_enclosed_mass[i] = np.sum(rho_psi[r_grid < R_half[i]]) * d.dx**3

			stellar_enclosed_mass[i] = np.sum(R < R_half[i]) * d.mp


	# first time the DM and stellar enclosed masses cross (None if they never do)
	time_when_equity = None
	if Show_T_Equality:
		diff = dm_enclosed_mass - stellar_enclosed_mass
		cross = np.where(diff[:-1] * diff[1:] <= 0)[0]
		if len(cross) > 0:
			k = cross[0]
			time_when_equity = brentq(lambda tt: np.interp(tt, t, diff), t[k], t[k + 1]) if diff[k] != diff[k + 1] else t[k]
		else:
			print("  no DM/stellar mass equality in [0, Tf]: M_DM(<R_half) - M_*(<R_half) goes from %.3g to %.3g" % (diff[0], diff[-1]))
	Show_line = Show_T_Equality and time_when_equity is not None


	if j == 0:
		fo.AddPlot(t, R_half, color = colors[j])
		#fo.AddLine(t, R_half_projected, color = colors[j], ls = '--', label = r'$R_\mathrm{half, projected}$ (2D)')

		if Show_line:
			fo.AddLine([time_when_equity, time_when_equity], [0, max_R*1.1], color = colors[j], ls = ':', label = r'$t_\mathrm{equality}$')
		#fo.AddLine(t, R_half_non_cent, color = colors[j], ls = '--', label = r'$R_\mathrm{half, noncent}$')
		# for i in range(len(R_parts[0])):
		# 	if i == 0:
		# 		fo.AddPlot(t, R_parts[:, i], color = colors[i])
		# 	else:
		# 		fo.AddLine(t, R_parts[:, i], color = colors[i])
				 
		fo2.AddPlot(t, plum_COM, color = colors[j])

	else:
		fo.AddLine(t, R_half, color = colors[j], label = labels[j])
		#fo.AddLine(t, R_half_non_cent, color = colors[j], ls = '--', label = r'$R_\mathrm{half, noncent}$')
		fo2.AddLine(t, plum_COM, color = colors[j], label = labels[j])
		#fo.AddLine(t, R_half_projected, color = colors[j], ls = '--', label = r'$R_\mathrm{half, projected}$ (2D)')
		if Show_line:
			fo.AddLine([time_when_equity, time_when_equity], [0, max_R*1.1], color = colors[j], ls = ':', label = r'$t_\mathrm{equality}$')

	return max_R



if __name__ == "__main__":
	fo = pu.FigObj()
	fo2 = pu.FigObj()

	for i in range(len(simNames)):
		print("Plotting " + simNames[i])
		d = do.MeshDataObj(simNames[i])
		max_R = PlotHalfLightRadius(d, fo, i, fo2, max_R, Show_T_Equality)

	fo.SetYLabel(r'$R$ [kpc]')
	fo.SetXLabel(r'$t$ [Myr]')
	fo.SetYLim(yLow = 0, yHigh = max_R * 1.1)
	fo.legend(fontsize = 12)
	fo.SetTitle(r'Half-light radius', fontsize = 18)
	fo.show()
	fo.save(d.dataDir + '/halflightradius')

	fo2.SetYLabel(r'$COM$')
	fo2.SetXLabel(r'$t$')
	fo2.legend(fontsize = 12)
	fo2.show()
	#fo2.save(d.dataDir + '/COM')
