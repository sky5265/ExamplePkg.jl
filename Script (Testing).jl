using Distributed

@everywhere begin
	using Pkg
	#Pkg.UPDATED_REGISTRY_THIS_SESSION[] = true
	Pkg.activate(".")
	#Pkg.instantiate()
	#Pkg.precompile()
end

@everywhere begin
	using Markdown
	using InteractiveUtils
	using FITSIO
	using DataFrames
	using LazyArrays, StructArrays
	using SpecialPolynomials
	using Gumbo
	using ThreadsX
	using Statistics: mean
	using FillArrays
	using StaticArrays
	using Polynomials
	using BenchmarkTools
	using Profile,  ProfileSVG, FlameGraphs
	using Plots # just needed for colorant when color FlameGraphs
	using LinearAlgebra, PDMats # used by periodogram.jl
	using Random
	using FLoops
	Random.seed!(123)
	using DistributedArrays
	using Distributions
	using SharedArrays
	using Test
end


#Testing Notebook
#In this notebook, we will run unit regression tests, where we run the serial and parallel implementations of line fitting in three different cases:
#1) Artifically created "perfect" lines, that have been created using Gauss-Hermite coefficients, so both implementations should perfectly fit to these lines. 
#2) Artficially created "terrible" lines. These will be rectangular lines, and the expectation is that both serial and parallel have a very hard time fitting to such  a shape. 
#In both cases, both serial and parallel should produce the same exact fits (as each other), so we will confirm that both arrive at the same answer. The test [TEST A] will be to chek if parallel and serial come to the same answer. 
#In case 1, the fits should be close to perfect, so the test [TEST B] will be to check that the loss function is very small. We can also test if the wavelength found by fitting is equal to the artificially created lambda [Test C]


# Importing Functions From Local File "


@everywhere include("./src/Support Functions.jl")
@everywhere using .SupportFunctions


# Perfect Lines Fitting to several lines and plotting a few of them

@everywhere artificial_σ = 0.04
@everywhere art_λs = [4500, 4500.3, 4500.6, 4500.9, 4501.2, 4501.5, 4501.8, 4502.1, 4502.4]


λ_to_plot_serial = []
hh_to_plot_serial = []
	
fitted_lines_serial = []
fitted_losses_serial = []
fitted_to_plot_serial = []


λ_to_plot_parallel = []
hh_to_plot_parallel = []
	
fitted_lines_parallel = []
fitted_losses_parallel = []
fitted_to_plot_parallel = []


for i in 1:length(art_λs)
	l_test = AbsorptionLine(art_λs[i], artificial_σ, (@SVector [-1*rand()/4, rand()/4, -1*rand()/4,0]))


	local_λ_serial, hh_serial, fitted0_serial = test_fit_perfect(l_test, 0)
	fitted_line_serial = fitted0_serial[1]
	loss_serial = fitted0_serial[2][1]

	push!(λ_to_plot_serial, local_λ_serial)
	push!(hh_to_plot_serial, hh_serial)

	push!(fitted_lines_serial, fitted_line_serial)
	push!(fitted_losses_serial, loss_serial)
	push!(fitted_to_plot_serial, fitted0_serial[1](local_λ_serial))


	local_λ_parallel, hh_parallel, fitted0_parallel = test_fit_perfect(l_test, 1)
	fitted_line_parallel = fitted0_parallel[1]
	loss_parallel = fitted0_parallel[2][1]

	push!(λ_to_plot_parallel, local_λ_parallel)
	push!(hh_to_plot_parallel, hh_parallel)

	push!(fitted_lines_parallel, fitted_line_parallel)
	push!(fitted_losses_parallel, loss_parallel)
	push!(fitted_to_plot_parallel, fitted0_parallel[1](local_λ_parallel))
end







	
plt = plot(legend = :bottomleft)
plot!(plt,λ_to_plot_serial[1],hh_to_plot_serial[1], label="Art. \'Perfect\' 1",color=:red)
plot!(plt,λ_to_plot_serial[1],fitted_to_plot_serial[1],label="Serial 1",color=:red)
plot!(plt,λ_to_plot_parallel[1],fitted_to_plot_parallel[1],label="Parallel 1",color=:red)

plot!(plt,λ_to_plot_serial[2],hh_to_plot_serial[2], label="Art. \'Perfect\' 2",color=:green)
plot!(plt,λ_to_plot_serial[2],fitted_to_plot_serial[2],label="Serial 2",color=:green)
plot!(plt,λ_to_plot_parallel[2],fitted_to_plot_parallel[2],label="Parallel 2",color=:green)

plot!(plt,λ_to_plot_serial[6],hh_to_plot_serial[6], label="Art. \'Perfect\' 3",color=:blue)
plot!(plt,λ_to_plot_serial[6],fitted_to_plot_serial[6],label="Serial 3",color=:blue)
plot!(plt,λ_to_plot_parallel[6],fitted_to_plot_parallel[6],label="Parallel 3",color=:blue)
savefig("Testing_Plots/Perfect_Lines_Fit.png")
	


#As you can see, when the artifically injected spectral line is a Gauss-Hermite function, the model finds it perfectly, and we cannot distinguish the model from the injected line. Let us run tests to show this is true."
#Displaying the fitted wavelength and loss of each fitted line
found_lines_serial = []
found_λs_serial = []

found_lines_parallel = []
found_λs_parallel = []

losses_serial = fitted_losses_serial
losses_parallel = fitted_losses_parallel
for i in 1:length(fitted_lines_serial)
	line_serial = fitted_lines_serial[i].lines[1]
	line_parallel = fitted_lines_parallel[i].lines[1]
	
	push!(found_lines_serial, line_serial)
	push!(found_lines_parallel, line_parallel)
	
	push!(found_λs_serial, line_serial.λ)
	push!(found_λs_parallel, line_parallel.λ)
end


@testset "Perfect Lines | Tests A, B, C" begin
	println("\n\n[PERFECT LINES]: The exact same fits are found by both serial and parallel (TEST A)")
	@test maximum(map(norm, fitted_to_plot_parallel.-fitted_to_plot_serial)) < 1E-4 * length(fitted_to_plot_parallel)
	
	println("[PERFECT LINES]: Testing that Loss is Small (TEST B)")
	@test maximum(map(norm, losses_serial.-losses_parallel)) < 1E-4 * length(losses_serial)
	
	println("[PERFECT LINES]: Comparing fitted wavelength to original (TEST C)")
	@test maximum(broadcast(abs, found_λs_serial.-art_λs)) < 1e-13
end;
println("\n\n")


# Terrible Lines | Creating several artificial lines and fitting to them	
vert_λs = [4500, 4500.5, 4501]

λ_to_plot_del = []
hh_to_plot_del = []
	
fitted_lines_del = []
fitted_losses_del = []
fitted_to_plot_del = []


λ_to_plot_del_parallel = []
hh_to_plot_del_parallel = []
	
fitted_lines_del_parallel = []
fitted_losses_del_parallel = []
fitted_to_plot_del_parallel = []



for i in 1:length(vert_λs)
	local_λ, hh, fitted_serial, fitted_parallel = test_fit_delta(vert_λs[i])
	
	fitted_line = fitted_serial[1]
	loss = fitted_serial[2][1]

	push!(λ_to_plot_del, local_λ)
	push!(hh_to_plot_del, hh)

	push!(fitted_lines_del, fitted_line)
	push!(fitted_losses_del, loss)
	push!(fitted_to_plot_del, fitted_serial[1](local_λ))


	
	
	fitted_line_parallel = fitted_parallel[1]
	loss_parallel = fitted_parallel[2][1]

	push!(λ_to_plot_del_parallel, local_λ)
	push!(hh_to_plot_del_parallel, hh)

	push!(fitted_lines_del_parallel, fitted_line_parallel)
	push!(fitted_losses_del_parallel, loss_parallel)
	push!(fitted_to_plot_del_parallel, fitted_parallel[1](local_λ))
	
end



	
plt = plot(legend = :bottomright)
plot!(plt,λ_to_plot_del[1],hh_to_plot_del[1], label="Rect. 1",color=:red)
plot!(plt,λ_to_plot_del[1],fitted_to_plot_del[1],label="Serial 1", color=:red)
plot!(plt,λ_to_plot_del_parallel[1],fitted_to_plot_del_parallel[1],label="Parallel 1", color=:red)

plot!(plt,λ_to_plot_del[2],hh_to_plot_del[2], label="Rect. 2",color=:purple)
plot!(plt,λ_to_plot_del[2],fitted_to_plot_del[2],label="Serial 2", color=:purple)
plot!(plt,λ_to_plot_del_parallel[2],fitted_to_plot_del_parallel[2],label="Parallel 2", color=:green)

plot!(plt,λ_to_plot_del[3],hh_to_plot_del[3], label="Rect. 3",color=:blue)
plot!(plt,λ_to_plot_del[3],fitted_to_plot_del[3],label="Serial 3", color=:blue)
plot!(plt,λ_to_plot_del_parallel[3],fitted_to_plot_del_parallel[3],label="Parallel 3", color=:blue)

plot!(plt, [4501.5], [0.00], color=:white, label="")

savefig("Testing_Plots/Terrible_Fits_Plot.png")


#Now, the artifically injected absorption line looks nothing like a Gauss-Hermite function and the fit cannot find it properly. 
#We can see the modeled value is quite different from the injected value. Let us check this using test cases


found_λs_del = []
found_λs_del_parallel = []

losses_del = fitted_losses_del
losses_del_parallel = fitted_losses_del_parallel
for i in 1:length(fitted_lines_del)
	line_del = fitted_lines_del[i].lines[1]
	line_del_parallel = fitted_lines_del_parallel[i].lines[1]

	push!(found_λs_del, line_del.λ)
	push!(found_λs_del_parallel, line_del_parallel.λ)
end


println("[TERRIBLE LINES]: The exact same fits are found by both serial and parallel (TEST A)")
println("[TERRIBLE LINES]: Testing that loss is large. Loss should be large, because it is a terrible fit.")
println("[TERRIBLE LINES]: Comparing fitted wavelength to original (TEST C) | Fitted wavelengths should be very different \nfrom original, which this test is looking for.")
@testset "Terrible Lines | Tests A, B, and C" begin
	@test maximum(map(norm, fitted_to_plot_del_parallel.-fitted_to_plot_del)) < 5E-5 * length(fitted_to_plot_del_parallel)
	
	# Checking if the largest loss value is small (TEST B) this should fail because it should be a terrible fit.
	@test maximum(losses_del) > 1E-5
	
	# Checking if the center of each line is found properly (TEST C)
	#this test should also fail-the algorithm may not find the correct central wavelength,
	#because the injected rectangular absorption line's dip is not symmetric about the central wavelength
	@test maximum(broadcast(abs, found_λs_del.-vert_λs)) > 1e-13
end;
println("\n\n")