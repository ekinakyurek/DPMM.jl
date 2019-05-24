function setup_visuals(X)
    if !isdefined(Main,:scene)
        @warn "setting up plots, takes a while for once"
        @eval Main using Makie
    end
    @eval Main scene  = scatter(X[1,:],X[2,:],color=DPMM.colorpalette[ones(Int,size(X,2))],markersize=0.5)
    @eval Main display(scene)
    colors = Main.scene[end][:color]
    axisnames = Main.scene[Main.Makie.Axis][:names][:axisnames]
    return (colors,axisnames)
end


function record!(observables::Any,z::AbstractArray,T::Int)
    z = first.(z)
    K=sort(unique(z))
    colors = map(zi->(findfirst(x->x==zi,K)-1)%12+1,z)
    observables[1][] = colorpalette[colors]
    observables[2][] = ("T=$T","")
    sleep(0.001)
end
record!(observables::Nothing,z::AbstractArray,T::Int) = nothing
