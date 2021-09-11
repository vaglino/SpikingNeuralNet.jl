# using LinearAlgebra, SparseArrays, LightGraphs
# using Plots,GraphRecipes

# function create_network(n_nodes= 100, k=10, β=0.1)

#     if k < log(n_nodes)
#         display("should be: n_nodes >> k >> ln(n_nodes)")
#     end
#      # create small-world network
#     g = watts_strogatz(n_nodes, k, β, is_directed=true)
#     adjacency = collect(adjacency_matrix(g))

#     return g, adjacency
# end

# # rows between 1 and n_inputs are by definition inputs
# g, adj = create_network(100, 10, 0.1)
# plot(g,curves=false)

# function initialize_lengths(adj,lengths=1:10)
#     len = adj .* rand(lengths,size(adj))
# end

# len = initialize_lengths(adj,1:10)

# function initialize_weights(adj; p_inhibition)
#     W = adj .* rand(Float64,size(adj))

#     # function inhibition(p_inhibition)
#         sign = (rand(Float64,size(adj)) .> p_inhibition) .*2 .-1
#         sign = sign .* adj
#     # end
#     W = W .* sign
# end

# W = initialize_weights(adj,p_inhibition=0.3)

# n = 100
# stepsize = 0.5
# spk = zeros(n,n)

# spk[1,:] .= adj[1,:] .* stepsize
# exc = zeros(n,n)

# function take_step(spk)
#     spk_mask = spk .!= 0 # find where there is a spike
#     spk += spk_mask * stepsize # advance by one step
# end

# function check_arrival(spk)
#     arrived = (spk .≈ len) .* adj
# end

# function synapse_integration(W, x, thresh)
#     # @show vec(sum( W .* x , dims = 1))
#     ŷ = activation.( vec(sum( W .* x , dims = 1)), threshold = thresh )

# end

# function activation(y; threshold=0.5)
#     ŷ = tanh(y)
#     firing = ŷ > threshold
# end


# function update_spikes(spk,exc_mask)
#     update = zeros(Float64,size(spk))
#     update[exc_mask,:] .= adj[exc_mask,:]
#     update[spk .!= 0] .= 0
#     spk = spk + update*stepsize
# end

# function delete_finished_spike(spk,arrived)
#     spk_mask = spk .!= 0
#     # spk_ind = index_mat(spk_mask)
#     spk[Bool.(arrived)] .= 0
#     return spk
# end

# function run_inference(spk; maxiters = 100)
#     excited_count = []
#     excited_id = []

#     # exc = zeros(n,n)
#     for i in 1:maxiters

#         spk = take_step(spk)

#         arrived = check_arrival(spk)

#         exc_mask = synapse_integration(W, arrived, thresh)

#         spk = update_spikes(spk,exc_mask)

#         spk = delete_finished_spike(spk,arrived)


#         # exc = exc*0

#         n_excited = sum(spk .!= 0)
#         push!(excited_count, n_excited)
#         push!(excited_id, exc_mask)
#     end
#     return excited_count, excited_id
# end

include("C:/Users/stravaglino3/Documents/net/dSNN.jl")
# parameters
n_neurons = 100
k = 10   # number of connections per neuron
β = 0.1  # probability of rewiring
n_inputs = 5

stepsize = 1.
thresh = 0.5
leak = 0.7

maxIters = 1000

# g, adj = create_network(n_neurons, 15, 0.1)
# #plot(g,curves=false)
#
# len = initialize_lengths(adj,1:10)
#
# W = initialize_weights(adj,p_inhibition=0.3)
#
#
# spk = zeros(Float64,size(adj))
#
# spk[1:n_inputs,:] .= adj[1:n_inputs,:] .* stepsize
# # exc = zeros(n,n)
#
#
#
# @time ncnt, id_excited = run_inference(spk,maxiters=maxIters)


g, adj = create_network(n_neurons, k, β)
#plot(g,curves=false)
len = initialize_lengths(adj,1:10)
W = initialize_weights(adj,p_inhibition=0.3)

spk = zeros(Float64,size(adj))
# spk[1:n_inputs,:] .= adj[1:n_inputs,:] .* stepsize

@time ncnt, id_excited = run_inference(spk,maxiters=maxIters)
plot_results()
loss(id_excited)
