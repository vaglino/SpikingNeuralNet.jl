function plot_results()
    p1 =  plot(ncnt)

    a = reduce(hcat,id_excited)
    a = Float64.(a)
    p2 = heatmap(a,seriescolor=:binary)

    hist = sum(a,dims=2)
    p3 = histogram(hist,bins=25)

    plot(p1, p2, layout=(1,2))
end

## Input handling
# given a normalized input signal strength ϵ [0,1], calculate
# how often the input neuron should spike. Finds the period between each spike

# signal strength to continuos periods
λ(s) = 1/s - 1 + t_step

# continuos periods to discrete periods
Λ(s) = floor( λ(s) / t_step ) * t_step

# max propagation velocity
v(ϵ,t_step) = ϵ / t_step

## loss function
function loss(id_excited,n_out)
    a = reduce(hcat,id_excited)
    a = Float64.(a)
    # sum(a[end-9:end],dims=2) ./ size(a,2)
    sum(a[end-n_out+1:end,:],dims=2) ./ size(a,2)
end
