# pylint: disable=C,W
import DataObj as do
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import numpy as np
import matplotlib.pyplot as plt

simName = "starTest/Dyn_friction/starTest_1e6_plummer_a0.77_Np_5e1_256"


def PlotVelHist(d):
	data_drops = d.data_drops

	drops = [0, data_drops // 3, data_drops // 2, 2 * data_drops // 3, data_drops]

	titles = [r'$t = 0$', r'$t = T/3$', r'$t = T/2$', r'$t = 2T/3$', r'$t = T$']

	fig, ax = plt.subplots(5, 1, figsize=(10, 12))

	for i, drop in enumerate(drops):
		r, v = d.LoadCorpData(drop)
		V = np.sqrt(np.sum(np.abs(v)**2, axis=1))
		logV = np.log10(V[V > 0])
		#logV = V
		ax[i].hist(logV, bins=50, density=True, range=(-4, -0.5))
		#ax[i].hist(logV, bins=50, density=True)
		tten_V = sorted(logV)[-10:]
		for j in range(10):
			ax[i].axvline(tten_V[j], color='r', linestyle='--', alpha=0.5, label='top 10 velocities' if j == 0 else "")
		ax[i].set_title(titles[i])
		ax[i].set_xlabel(r'$\log_{10}(v)$')
		ax[i].set_ylabel(r'$P(\log_{10}(v))$')
		ten_percentile = np.percentile(logV, 90)
		five_percentile = np.percentile(logV, 95)
		two_percentile = np.percentile(logV, 98)
		one_percentile = np.percentile(logV, 99)
		ax[i].axvline(ten_percentile, color='g', linestyle='--', alpha=0.5, label='90th percentile')
		ax[i].axvline(five_percentile, color='b', linestyle='--', alpha=0.5, label='95th percentile')
		ax[i].axvline(two_percentile, color='c', linestyle='--', alpha=0.5, label='98th percentile')
		ax[i].axvline(one_percentile, color='m', linestyle='--', alpha=0.5, label='99th percentile')
		a_stars = 0.1
		M_stars = 1e6
		v_esc = np.sqrt(2 * au.G * M_stars) * (a_stars**2)**(-1 / 4)
		ax[i].axvline(np.log10(v_esc), color='k', linestyle='--', alpha=1, label='Escape velocity')
		ax[i].legend()


if __name__ == "__main__":

	d = do.MeshDataObj(simName)
	PlotVelHist(d)

	plt.tight_layout()
	plt.savefig(d.dataDir + 'velocity_histograms.png', dpi=300)
	plt.show()
