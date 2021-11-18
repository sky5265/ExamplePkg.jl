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
	Random.seed!(123)
	using Distributions
	using SharedArrays
	using FLoops
	end;
	
#addprocs(4)



@everywhere include("./src/Support Functions.jl")
@everywhere using .SupportFunctions


@everywhere data_path = ""  #Data is stored in same directory as this file
@everywhere filename = "neidL1_20210305T170410.fits"
@everywhere f = FITS(joinpath(data_path,filename))
@everywhere order_idx = 40      # Arbitrarily picking just one order for this example
@everywhere pix_fit = 1025:8096  # Exclude pixels near edge of detector where there are issues that complicate things and we don't want to be distracted by



#reading in wavelength, flux, and variance data
@everywhere λ_raw = read(f["SCIWAVE"],:,order_idx)
@everywhere flux_raw = read(f["SCIFLUX"],:,order_idx)
@everywhere var_raw = read(f["SCIVAR"],:,order_idx)


#cutting out nans from dataset
@everywhere mask_nonans = findall(.!(isnan.(λ_raw) .| isnan.(flux_raw) .| isnan.(var_raw) ))
@everywhere λ = λ_raw[mask_nonans]
@everywhere flux = flux_raw[mask_nonans]
@everywhere var = var_raw[mask_nonans]



# Removing Blaze background by dividing observations by blaze function
	
	
@everywhere mask_fit = pix_fit[findall(.!(isnan.(λ_raw[pix_fit]) .| isnan.(flux_raw[pix_fit]) .| isnan.(var_raw[pix_fit]) ))]
@everywhere blaze_model0 =  fit_blaze_model(1:length(λ),flux,var,order=8, mask=mask_fit)
@everywhere pix_gt_100p_it1 = flux[pix_fit]./blaze_model0.(pix_fit) .>= 1.0

@everywhere blaze_model1 =  fit_blaze_model( (1:length(λ)), flux, var,  order=8, mask=pix_fit[pix_gt_100p_it1])
@everywhere pix_gt_100p_it2 = flux[pix_fit]./blaze_model1.(pix_fit) .>= 1.0

@everywhere blaze_model2 =  fit_blaze_model( (1:length(λ)), flux, var,  order=8, mask=pix_fit[pix_gt_100p_it2])


	
@everywhere pix_plt = closest_index(λ, 4550):closest_index(λ, 4610) #finds the indices that most closely correspond to λ = 4569.94 and λ=4579.8 angstroms. These wavelengths are mostly just arbitrarily chosen, where we found lines within this wavelength band from a list of known absorption lines in the Sun. When we move to parallelized code, we will look at all wavelengths, not just this smaller band.


@everywhere df = DataFrame( λ=view(λ,pix_plt), 
			flux=view(flux,pix_plt)./blaze_model2.(pix_plt),
			var =view(var,pix_plt)./(blaze_model2.(pix_plt)).^2
			)

num_gh_orders = 4 #here we can set the order of GH polynomial fits for the rest of the code


#Found these absorption lines with these standard deviations for each line. 
#λ_lines = [ 4555.3,4559.95, 4563.23989931, 4563.41910949, 4563.76449209, 4564.17151481, 4564.34024273, 4564.69744168, 4564.82724255, 4565.5196282, 4565.66471622, 4566.23229193, 4566.51949452, 4566.87124681, 4567.40957888, 4568.32947197, 4568.60675666, 4568.77955633, 4569.35589751, 4569.61424051, 4570.02161725, 4570.37846574, 4571.09895901, 4571.43706403, 4571.67596248, 4571.97850945, 4572.27601199, 4572.60413885, 4572.86475639, 4573.80826847, 4573.9777376, 4574.22058447, 4574.47278377, 4574.72227641, 4575.10790085, 4575.54216767, 4575.78807278, 4576.33735793, 4577.17797639, 4577.48371817, 4577.69571021, 4578.03225993, 4578.32482976, 4578.55743589, 4579.05844195, 4579.32999314, 4579.51175328, 4579.67461995, 4579.81960761, 4580.05636844, 4580.41671944, 4580.5881527, 4581.04111864, 4581.20192551, 4582.30743629, 4582.49483022, 4582.83395544, 4583.12727938, 4583.41373121, 4583.83832133, 4584.28107653, 4584.81731555, 4585.08049157, 4585.34095137, 4585.8742966, 4586.22544989, 4586.3712534, 4587.13181633, 4587.72177531, 4588.20292127, 4588.68796051, 4589.01082716, 4589.29840078, 4589.95113751, 4590.7903537, 4591.110535, 4591.39612609, 4592.05460424, 4592.36460681, 4592.65733244, 4593.17249527, 4593.52895394, 4593.92276107, 4594.11902026, 4594.41880008, 4594.63251674, 4594.89747512, 4595.36207176, 4595.59618168, 4596.0598587, 4596.41240259, 4596.57597596, 4596.90736432, 4597.24435651, 4597.38289242, 4597.75193956, 4597.86998627, 4598.12294754, 4598.37433123, 4598.74391477, 4598.99676453, 4599.22776442, 4599.84019949] #length 103
@everywhere λ_lines_eyeball = [4550.0, 4552.042, 4553.75, 4555.3,4556.76, 4557.2, 4557.4, 4561.341, 4562.63, 4565, 4566.87124681,  4567.69, 4568.3, 4569.5, 4570, 4570.8, 4572.4, 4573.4, 4575.39, 4575.9, 4577.5, 4578.58, 4580, 4581.4, 4581.8, 4582.7, 4584.01, 4585, 4586, 4587.1, 4587.5, 4588.53, 4589.5, 4591.3, 4592.6, 4593.475, 4594, 4595.28, 4596.5, 4597.4, 4599, 4599.5, 4601, 4601.5, 4602, 4603.18, 4604.1, 4605.9, 4606.2, 4606.8, 4607.63, 4608.5, 4609] # has length 53
@everywhere λ_lines = λ_lines_eyeball




# Benchmarking

fit_lines_v0_serial(λ_lines,df.λ,df.flux,df.var, order = 
num_gh_orders)
println("Serial Run Benchmark:\n")
fitted_timing_serial = @btime fit_lines_v0_serial(λ_lines,df.λ,df.flux,df.var, order = num_gh_orders)
#println(fitted_timing_serial)
println("\n\n")


fit_lines_v0_parallel_experimental(λ_lines,df.λ,df.flux,df.var, order = num_gh_orders)
println("Parallel Run Benchmark:\n")
timing_parallel = @btime fit_lines_v0_parallel_experimental(λ_lines,df.λ,df.flux,df.var, order = num_gh_orders)
#println(timing_parallel)
println("\n\n")