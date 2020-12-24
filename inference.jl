include("dSNN.jl")

# parameters
n_neurons = 100
k = 10   # number of connections per neuron
β = 0.1  # probability of rewiring
n_inputs = 5

stepsize = 1.
thresh = 0.6
leak = 0.7

t_max = 1000

g, adj = create_network(n_neurons, k, β)
#plot(g,curves=false)
len = initialize_lengths(adj,1:10)
W = initialize_weights(adj,p_inhibition=0.3)

spk = zeros(Float64,size(adj))
spk[1:n_inputs,:] .= adj[1:n_inputs,:] .* stepsize

@time ncnt, id_excited = run_inference(spk,maxiters=t_max)
plot_results()
loss(id_excited)
