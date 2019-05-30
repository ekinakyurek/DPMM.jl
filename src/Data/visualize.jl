"""

    setup_scene(X)

    Initialize plots for visualizing 2D data
"""
function setup_scene(X)
    if !isdefined(Main,:scene)
        @warn "setting up the plot, takes a while for once"
        @eval Main using Makie
    end
    @eval Main scene = scatter($(X[1,:]),$(X[2,:]),color=DPMM.colorpalette[ones(Int,$(size(X,2)))],markersize=0.1)
    @eval Main display(scene)
    return Main.scene
end


function record!(scene::Any,z::AbstractArray,T::Int)
    z = first.(z)
    K=sort(unique(z))
    colors = map(zi->(findfirst(x->x==zi,K)-1)%12+1,z)
    scene[end][:color][] = colorpalette[colors]
    scene[Main.Makie.Axis][:names][:axisnames][] = ("T=$T","")
    sleep(0.001)
end
record!(scene::Nothing,z::AbstractArray,T::Int) = nothing
