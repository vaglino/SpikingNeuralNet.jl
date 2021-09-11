include("dSNN.jl")

# parameters
n_neurons = 30
k = 10   # number of connections per neuron
β = 0.1  # probability of rewiring
n_inputs = 2

t_step = 1.0
thresh = 0.5
leak = 0.7

t_max = 1000

g, adj = create_network(n_neurons, k, β)
#plot(g,curves=false)
len = initialize_lengths(adj,1:0.1:10.)
W = initialize_weights(adj,p_inhibition=0.3)

# vmax = v(1.,1.)
vmax = v(0.1,t_step)


spk = zeros(Float64,size(adj))
inputs = collect(range(0.0,stop=0.7,length=n_inputs))
Λs = Λ.(inputs)

@time ncnt, id_excited = run_inference(spk,maxiters=t_max)
plot_results()

n_out = 2
out = loss(id_excited,n_out)
ŷ = out ./ sum(out)

using Flux
ŷ = [0.99,0.01]
y = [0,1]
Flux.Losses.logitcrossentropy(ŷ, y)

using ReverseDiff, Flux, Zygote

function lossfunction(W)
    ncnt, id_excited = run_inference(spk,maxiters=t_max)
    n_out = 2
    out = loss(id_excited,n_out)

    ŷ = out ./ sum(out)
    @show ŷ
    # Flux.Losses.logitcrossentropy(ŷ, y)
    Flux.Losses.mse(ŷ, y)
end

y = [0,1]
lossfunction(W)

gw = W->ReverseDiff.gradient(lossfunction,W)
gr = gw(W)
sum(gr)
Zygote.gradient(lossfunction,W)
