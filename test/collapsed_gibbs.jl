using DPMM, Makie

gmodel = GridMixture(6)
X,labels = rand_with_label(gmodel,1000)
scene  = scatter(X[1,:],X[2,:],color=DPMM.colorpalette[0labels.+1],markersize=1.0)
colors = scene[end][:color]
axisnames = scene[Axis][:names][:axisnames]
quasi_collapsed_gibbs(X,T=10,observables=(colors,axisnames))
collapsed_gibbs(X,T=10,observables=(colors,axisnames))
