
t_step = 0.5
s = 0.1:0.001:1

λs = λ.(s)
Λs = Λ.(s)

fig = plot(s,λs,label="λs")
plot!(s,Λs,label="Λs")
plot!([0,1.0], [t_step,t_step], label="t_step",linestyle = :dash, color= :grey)
ylims!((0,Inf))
title!("Input function for t_step = $t_step")


filename = "input_fn_t_step_0_5)"
savefig(fig, filename)
