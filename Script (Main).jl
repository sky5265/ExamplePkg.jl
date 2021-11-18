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
	#using PlutoUI, PlutoTest, PlutoTeachingTools
	using BenchmarkTools
	using Profile,  ProfileSVG, FlameGraphs
	using Plots 
	using LinearAlgebra, PDMats # used by periodogram.jl
	using Random
	Random.seed!(123)
	using FLoops
	using Test
end


@everywhere include("./src/Support Functions.jl")
@everywhere using .SupportFunctions



data_path = ""  #Data is stored in same directory as this file
filename = "neidL1_20210305T170410.fits"   
f = FITS(joinpath(data_path,filename))
order_idx = 40      # Arbitrarily picking just one order for this example
pix_fit = 1025:8096  # Exclude pixels near edge of detector where there are issues that complicate things and we don't want to be distracted by



#reading in wavelength, flux, and variance data
λ_raw = read(f["SCIWAVE"],:,order_idx)
flux_raw = read(f["SCIFLUX"],:,order_idx)
var_raw = read(f["SCIVAR"],:,order_idx)


#cutting out nans from dataset
mask_nonans = findall(.!(isnan.(λ_raw) .| isnan.(flux_raw) .| isnan.(var_raw) ))
λ = λ_raw[mask_nonans]
flux = flux_raw[mask_nonans]
var = var_raw[mask_nonans]



@testset "Testing input data was read correctly" begin

	@test length(λ) > 0
	@test length(flux) == length(λ) == length(var)
end



# Removing Background Diffraction Pattern
# When light passes through a diffraction grating, it takes the shape of a Blaze function, so the observations follow a Blaze shape. 
# We remove the Blaze background here by fitting a model (called Blaze 2) and removing that from the data.
# Removing Blaze background by dividing observations by blaze function
	

mask_fit = pix_fit[findall(.!(isnan.(λ_raw[pix_fit]) .| isnan.(flux_raw[pix_fit]) .| isnan.(var_raw[pix_fit]) ))]
blaze_model0 =  fit_blaze_model(1:length(λ),flux,var,order=8, mask=mask_fit)
pix_gt_100p_it1 = flux[pix_fit]./blaze_model0.(pix_fit) .>= 1.0

blaze_model1 =  fit_blaze_model( (1:length(λ)), flux, var,  order=8, mask=pix_fit[pix_gt_100p_it1])
pix_gt_100p_it2 = flux[pix_fit]./blaze_model1.(pix_fit) .>= 1.0

blaze_model2 =  fit_blaze_model( (1:length(λ)), flux, var,  order=8, mask=pix_fit[pix_gt_100p_it2])

p = plot()
plot!(p, λ[pix_fit],flux[pix_fit], label="Observation",color=:grey)
plot!(p, λ[pix_fit],blaze_model2.(pix_fit), label="Blaze 2",color=:blue)
xlabel!("λ (Å)")
ylabel!("Flux")
savefig("Main Outputs/Fitted_Blaze.png")
	


pix_fit1 = closest_index(λ, 4550):closest_index(λ, 4560)
p1 = plot(legend=:bottomright)
plot!(p1, λ[pix_fit],flux[pix_fit]./blaze_model2.(pix_fit), label="Observation/Blaze 2", color=:blue, legend=:bottomright)
xlabel!("λ (Å)")
ylabel!("Normalized Flux")
savefig("Main Outputs/Blaze_Removed.png")


#Picking the full range of wavelengths. 



pix_plt = closest_index(λ, 4550):closest_index(λ, 4610) #finds the indices that most closely correspond to λ = 4569.94 and λ=4579.8 angstroms. These wavelengths are mostly just arbitrarily chosen, where we found lines within this wavelength band from a list of known absorption lines in the Sun. When we move to parallelized code, we will look at all wavelengths, not just this smaller band.

# Building a dataframe, where the fluxes have the blaze background removed
df = DataFrame( λ=view(λ,pix_plt), 
			flux=view(flux,pix_plt)./blaze_model2.(pix_plt),
			var =view(var,pix_plt)./(blaze_model2.(pix_plt)).^2
			)
	



# Fit to Real Solar Absorption Lines"
# Note: we use a loss function to evaluate the fitting of each line. Loss function is defined as sum[abs(predicted-actual flux)] for λ = λ-3*σ:λ-3*σ



num_gh_orders = 4 #here we can set the order of GH polynomial fits for the rest of the code


#Found these absorption lines with these standard deviations for each line. 
#λ_lines = [ 4555.3,4559.95, 4563.23989931, 4563.41910949, 4563.76449209, 4564.17151481, 4564.34024273, 4564.69744168, 4564.82724255, 4565.5196282, 4565.66471622, 4566.23229193, 4566.51949452, 4566.87124681, 4567.40957888, 4568.32947197, 4568.60675666, 4568.77955633, 4569.35589751, 4569.61424051, 4570.02161725, 4570.37846574, 4571.09895901, 4571.43706403, 4571.67596248, 4571.97850945, 4572.27601199, 4572.60413885, 4572.86475639, 4573.80826847, 4573.9777376, 4574.22058447, 4574.47278377, 4574.72227641, 4575.10790085, 4575.54216767, 4575.78807278, 4576.33735793, 4577.17797639, 4577.48371817, 4577.69571021, 4578.03225993, 4578.32482976, 4578.55743589, 4579.05844195, 4579.32999314, 4579.51175328, 4579.67461995, 4579.81960761, 4580.05636844, 4580.41671944, 4580.5881527, 4581.04111864, 4581.20192551, 4582.30743629, 4582.49483022, 4582.83395544, 4583.12727938, 4583.41373121, 4583.83832133, 4584.28107653, 4584.81731555, 4585.08049157, 4585.34095137, 4585.8742966, 4586.22544989, 4586.3712534, 4587.13181633, 4587.72177531, 4588.20292127, 4588.68796051, 4589.01082716, 4589.29840078, 4589.95113751, 4590.7903537, 4591.110535, 4591.39612609, 4592.05460424, 4592.36460681, 4592.65733244, 4593.17249527, 4593.52895394, 4593.92276107, 4594.11902026, 4594.41880008, 4594.63251674, 4594.89747512, 4595.36207176, 4595.59618168, 4596.0598587, 4596.41240259, 4596.57597596, 4596.90736432, 4597.24435651, 4597.38289242, 4597.75193956, 4597.86998627, 4598.12294754, 4598.37433123, 4598.74391477, 4598.99676453, 4599.22776442, 4599.84019949] #length 103
λ_lines_eyeball = [4550.0, 4552.042, 4553.75, 4555.3,4556.76, 4557.2, 4557.4, 4561.341, 4562.63, 4565, 4566.87124681,  4567.69, 4568.3, 4569.5, 4570, 4570.8, 4572.4, 4573.4, 4575.39, 4575.9, 4577.5, 4578.58, 4580, 4581.4, 4581.8, 4582.7, 4584.01, 4585, 4586, 4587.1, 4587.5, 4588.53, 4589.5, 4591.3, 4592.6, 4593.475, 4594, 4595.28, 4596.5, 4597.4, 4599, 4599.5, 4601, 4601.5, 4602, 4603.18, 4604.1, 4605.9, 4606.2, 4606.8, 4607.63, 4608.5, 4609] # has length 53
λ_lines = λ_lines_eyeball


#Fitting in serial to these lines
fitted0 = fit_lines_v0_parallel_experimental(λ_lines,df.λ,df.flux,df.var, order = num_gh_orders)
fitted_lines = fitted0[1].lines	
losses0 = fitted0[2]
	


@testset "Testing losses were properly found" begin

	@test length(losses0) == length(fitted_lines) == length(λ_lines) #testing to make sure all the lines have been fitted to, and there is a loss for each fit
end


# Plotting all fitted lines

fit_windows = fill(0.0,2*length(λ_lines))
loss_windows = fill(0.0,2*length(λ_lines))

N = length(λ_lines)
for i in 1:N
	σ_h = 0.04
	fit_windows[2*(i)-1] = λ_lines[i]-fit_devs*σ_h
	loss_windows[2*(i)-1] = λ_lines[i]-loss_devs*σ_h
	
	fit_windows[2*(i)] = λ_lines[i]+fit_devs*σ_h
	loss_windows[2*(i)] = λ_lines[i]+loss_devs*σ_h
end


#local plt = plot(legend=:bottomright)
p2 = plot()
plot!(p2,df.λ,df.flux, label="Observation/Fit")
plot!(p2,df.λ,fitted0[1](df.λ),label="Model")
#vline!(λ_lines, label="Line Centers")
#vline!(fit_windows, label="Fit window")
#vline!(loss_windows, label="Loss window")
xlabel!("λ (Å)")
ylabel!("Normalized Flux")
savefig("Main Outputs/Fitted Lines.png")




# Printing out Losses, Writing to file
println(losses0)
losses_file_out = open("Main Outputs/fitted_losses.txt", "w")
for i in 1:length(losses0)
	write(losses_file_out, losses0[i])
	write(losses_file_out, "\n")
end


# Printing fitted lines`
println(fitted_lines)
fitted_lines_out = open("Main Outputs/fitted_lines.txt", "w")
for i in 1:length(fitted_lines)
	write(fitted_lines_out, fitted_lines[i].gh_coeff)
	write(fitted_lines_out, "\n")
end



#reading Gauss-Hermite Coefficients from the fitted lines

found_gh_output = open("Main Outputs/ghs_out.txt", "w")
	
num_lines_found = length(fitted_lines)
num_gh_coeff = length(fitted_lines[1].gh_coeff)
gh_s = reshape(zeros(num_lines_found*num_gh_coeff), num_lines_found, num_gh_coeff)
#gh_s = Array{Float64}(undef, length(lines_found), length(lines_found[1].gh_coeff))
for i in 1:length(fitted_lines)
	gh = fitted_lines[i].gh_coeff
	for j in 1:length(gh)
		write(found_gh_output, gh[j])
		write(found_gh_output, " ")
		gh_s[i,j] = gh[j]
	end
	write(found_gh_output, "\n")
end



# Running tests on the found Gauss-Hermite Coefficients
@testset "Confirming Gauss-Hemite Coefficients Properly Found" begin
@test num_lines_found == length(λ_lines) #make sure all lines have been fitted to
@test num_gh_coeff == num_gh_orders #make sure all orders exist
#checking if all lines have exactly the same number of gh_coefficients
begin
	all_same_size = 1 #1 as in true, if false, it will be 0
	for i in 1:length(fitted_lines)
		if (length(fitted_lines[i].gh_coeff) != num_gh_coeff)
			all_same_size = 0
		end
	end
	@test all_same_size == 1
end

end