using DPMM, Makie

gmodel = GridMixture(3)
X,labels = rand_with_label(gmodel,10000)
scene  = scatter(X[1,:],X[2,:],color=DPMM.colorpalette[0labels.+1],markersize=0.05)
colors = scene[end][:color]
axisnames = scene[Axis][:names][:axisnames]
quasi_direct_gibbs(X,ninit=9,T=10,observables=(colors,axisnames))
direct_gibbs(X,T=10,ninit=9,observables=(colors,axisnames))
