module SupportFunctions

	using StaticArrays
	using Distributed
	#using Distributions
	using StaticArrays
	using SharedArrays
	using Polynomials
	using FillArrays
	using Statistics
	using ThreadsX
	using Gumbo
	using FITSIO
	using DataFrames
	using LazyArrays, StructArrays
	using SpecialPolynomials
	using LinearAlgebra, PDMats
	using FLoops
	using Plots

	export test, fit_lines_v0_serial, fit_lines_v0_parallel, closest_index, gaussian, gauss_hermite_basis, SpectrumModel, AbsorptionLine, gh_polynomials,  BlazeModel, fit_blaze_model, fit_blaze_model_v0, fit_devs, loss_devs, test_fit_perfect, test_fit_delta, fit_lines_v0_parallel_experimental

	#Blaze Function Modeling


	function test()
		println("Hello world")
	end

	struct BlazeModel{T}
		x_mean::T
		polyn::Polynomial{T,:x}
	end
	
	function (blaze::BlazeModel{T})(x::T1)   where { T<:Number, T1<:Number } 
		blaze.polyn(x-blaze.x_mean)
	end



	function fit_blaze_model_v0(x::V1, flux::V2, var::V3,
							T::Type = promote_type(T1,T2,T3); order::Integer = 8, mask = 1:length(x)  ) where
				{ T1<:Number, T2<:Number, T3<:Number,
				V1<:AbstractVector{T1}, V2<:AbstractVector{T2}, V3<:AbstractVector{T3} } 
		x_mean = mean(x[mask])
		fit_result = fit( x[mask].-x_mean, 
						  flux[mask], order, weights=1.0./var[mask] )
		return BlazeModel{T}(x_mean,fit_result)
	end;

	function fit_blaze_model(x::V1, flux::V2, var::V3,
							T::Type = promote_type(T1,T2,T3); order::Integer = 8, mask = 1:length(x)  ) where
				{ T1<:Number, T2<:Number, T3<:Number,
				V1<:AbstractVector{T1}, V2<:AbstractVector{T2}, V3<:AbstractVector{T3} } 
		x_mean = mean(view(x,mask))
		fit_result = fit( view(x,mask).-x_mean,
						  view(flux,mask), order, weights=1.0./view(var,mask) )
		return BlazeModel{T}(x_mean,fit_result)
	end;



	#Absorption Line and GH Parametrization of it

	struct AbsorptionLine
			λ::Float64
			σ::Float64
			gh_coeff::SVector{4,Float64}  # Statically allocated to reduce memory allocations
		end;

	function (line::AbsorptionLine)(λ)
			T = typeof(line.λ)
			gh_polys::Vector{Polynomial{T,:x}} = gh_polynomials 
			x = (λ.-line.λ)./line.σ
			
			g = exp.(-0.5.*x.^2) 
			# Could eliminate some allocations in mapreduce with hcat.
			p = mapreduce(i->line.gh_coeff[i].*gh_polys[i].(x),hcat,1:length(line.gh_coeff))
			one(λ) .+ sum(g.*p)
			#sum(g.*p)
		end

	function gaussian(line::AbsorptionLine, λ::Number)
			exp(-((λ-line.λ)/line.σ)^2//2)
		end
		
	# Precompute Gauss-Hermite polynomials once.
	# Should move to module, so not in global scope
	gh_polynomials = [basis(Hermite, i)(variable(Polynomial{Float64})) for i in 0:10]

		
	function gauss_hermite_basis(line::AbsorptionLine, λ, order::Integer)
		@assert 1 <= order+1 <= length(gh_polynomials)
		T = typeof(line.λ)
		gh_poly::Polynomial{T,:x} = gh_polynomials[order+1] 
		x = (λ.-line.λ)./line.σ
		exp.(-0.5.*x.^2) .* gh_poly.(x)		
	end




	function gauss_hermite_basis(line::AbsorptionLine, λ; orders::AbstractVector{Int64} = 1:length(line.gh_coeff) )
		T = typeof(line.λ)
		gh_polys::Vector{Polynomial{T,:x}} = gh_polynomials
		x = (λ.-line.λ)./line.σ
		g = exp.(-0.5.*x.^2) 
		p = mapreduce(i->gh_polys[i].(x),hcat,orders)
		g.*p
	end




	#Synthetic Spectrum Definitions

	struct SpectrumModel
		norm::Float64
		lines::Vector{AbsorptionLine}
	end

	function (model::SpectrumModel)(λ)
		result = fill(model.norm,length(λ))
		for i in 1:length(model.lines)
			result .*= model.lines[i].(λ)
		end
		return result
	end




	#Fitting Stuff




	function closest_index(λ::V1, num_to_find::Number) where {T1<:Number, V1<:AbstractVector{T1}}
		#this function finds the index in the array λ of the value that is closest to num_to_find
		max_val = maximum(λ)
		min_val = minimum(λ)
		i_tor = 1 #"tor" as in "to return"
		min_diff = (max_val-min_val)
		for i in eachindex(λ)
			curr_diff = abs(λ[i] - num_to_find)
			if curr_diff < min_diff
				i_tor = i
				min_diff = curr_diff
			end
		end
		@assert i_tor >= 0 #make sure that the function isn't somehow magically returning a negative index
		@assert i_tor <= length(λ) #make sure the function doesn't return an index that is larger than the arrya
		return i_tor
	end;


	loss_devs = 5
	fit_devs = 3



	function fit_lines_v0_serial(λ_lines::V1, λ::V3, flux::V4, var::V5, T::Type = promote_type(T1,T3,T4,T5); order::Integer ) where
				{ T1<:Number, T3<:Number, T4<:Number, T5<:Number,
				  V1<:AbstractVector{T1}, V3<:AbstractVector{T3}, V4<:AbstractVector{T4}, V5<:AbstractVector{T5}  } 
		
		#fits GH polynomials to an array of absorption lines, taking in an array of λs to fit at and a corresponding σ values for each absorption line. This returns a list of fitted lines (each has the num_gh_orders-ordered GH fit) and a list of losses for each fit, evaluating the fit.
		
		@assert size(λ) == size(flux) == size(var)
		n_pix = length(λ)
		n_lines = length(λ_lines)
		n = length(λ)
		@assert n_lines >= 1 
		@assert 1 <= order <= 10 #10 is the number of gh_polynomials I have computed in the global scope. The code cannot handle more than these many (although we will seldom go beyond 4 orders)
		@assert n_pix > 1 + order*n_lines  # number of fit parameters
		covar = PDiagMat(var)   # diagonal covariance matrix
		design_matrix = ones(n_pix,1)
		
		fitted_lines = []
		fitted_losses = []
		for i in 1:n_lines
			λ_line = λ_lines[i]
			σ_line = 0.04
			#will try fitting to every wavelength in the range λ_line-0.5*σ_line to λ_line+0.5*σ_line
			lowest_idx = closest_index(λ, λ_line-fit_devs*σ_line)
			highest_idx = closest_index(λ, λ_line+fit_devs*σ_line)
			
			losses = zeros(highest_idx-lowest_idx+1)
			line_tries = []
			for j = lowest_idx:highest_idx
				λ_to_fit = λ[j]

				line = AbsorptionLine(λ_to_fit, σ_line,(@SVector zeros(order)) ) #create a line data-structure 
				
				design_matrix = hcat(ones(n),		gauss_hermite_basis(line,λ,orders=1:order)  )  	# fit to the line	
				Xt_inv_covar_X = design_matrix' * (covar \ design_matrix) 
				X_inv_covar_y =   design_matrix' * (covar \ flux ) 
				coeff_hat = (Xt_inv_covar_X \ X_inv_covar_y)
				
				
				line = AbsorptionLine(λ_to_fit, σ_line, coeff_hat[2:end] )
				
				push!(line_tries, line)
				#line_tries.append(line)
				#calculating loss for this fit
				lowest_loss_idx = closest_index(λ, λ_line-loss_devs*σ_line)
				highest_loss_idx = closest_index(λ, λ_line+loss_devs*σ_line)
				loss = 0.0
				for k = lowest_loss_idx:highest_loss_idx
					loss+=abs(flux[k]-line.(λ[k]))
				end
				losses[j-lowest_idx+1] = loss
			end
			
			#find fit with lowest loss
			best_fit_loss, best_fit_idx = findmin(losses)
			best_fit_λ = λ[best_fit_idx]
			best_fit_line = line_tries[best_fit_idx]
			
			push!(fitted_lines, best_fit_line)
			push!(fitted_losses, best_fit_loss)
		end
		return SpectrumModel(1,fitted_lines), fitted_losses
		
	end;


	function fit_lines_v0_parallel(λ_lines::V1, λ::V3, flux::V4, var::V5, T::Type = promote_type(T1,T3,T4,T5); order::Integer ) where
				{ T1<:Number, T3<:Number, T4<:Number, T5<:Number,
				  V1<:AbstractVector{T1}, V3<:AbstractVector{T3}, V4<:AbstractVector{T4}, V5<:AbstractVector{T5}  } 
		
		#fits GH polynomials to an array of absorption lines, taking in an array of λs to fit at and a corresponding σ values for each absorption line. This returns a list of fitted lines (each has the num_gh_orders-ordered GH fit) and a list of losses for each fit, evaluating the fit.
		
		@assert size(λ) == size(flux) == size(var)
		n_pix = length(λ)
		n_lines = length(λ_lines)
		n = length(λ)
		@assert n_lines >= 1 
		@assert 1 <= order <= 10 #10 is the number of gh_polynomials I have computed in the global scope. The code cannot handle more than these many (although we will seldom go beyond 4 orders)
		@assert n_pix > 1 + order*n_lines  # number of fit parameters
		covar = PDiagMat(var)   # diagonal covariance matrix
		design_matrix = ones(n_pix,1)
		
		fitted_lines = []
		fitted_losses = []
		for i in 1:n_lines
			λ_line = λ_lines[i]
			σ_line = 0.04
			#will try fitting to every wavelength in the range λ_line-0.5*σ_line to λ_line+0.5*σ_line
			lowest_idx = closest_index(λ, λ_line-fit_devs*σ_line)
			highest_idx = closest_index(λ, λ_line+fit_devs*σ_line)
			
			losses = zeros(highest_idx-lowest_idx+1)
			line_tries = []
			for j = lowest_idx:highest_idx
				λ_to_fit = λ[j]

				line = AbsorptionLine(λ_to_fit, σ_line,(@SVector zeros(order)) ) #create a line data-structure 
				
				design_matrix = hcat(ones(n),		gauss_hermite_basis(line,λ,orders=1:order)  )  	# fit to the line	
				Xt_inv_covar_X = design_matrix' * (covar \ design_matrix) 
				X_inv_covar_y =   design_matrix' * (covar \ flux ) 
				coeff_hat = (Xt_inv_covar_X \ X_inv_covar_y)
				
				
				line = AbsorptionLine(λ_to_fit, σ_line, coeff_hat[2:end] )
				
				push!(line_tries, line)
				#line_tries.append(line)
				#calculating loss for this fit
				lowest_loss_idx = closest_index(λ, λ_line-loss_devs*σ_line)
				highest_loss_idx = closest_index(λ, λ_line+loss_devs*σ_line)
				loss = 0.0
				Threads.@threads for k = lowest_loss_idx:highest_loss_idx
					loss+=abs(flux[k]-line.(λ[k]))
				end
				losses[j-lowest_idx+1] = loss
			end
			
			#find fit with lowest loss
			best_fit_loss, best_fit_idx = findmin(losses)
			best_fit_λ = λ[best_fit_idx]
			best_fit_line = line_tries[best_fit_idx]
			
			push!(fitted_lines, best_fit_line)
			push!(fitted_losses, best_fit_loss)
		end
		return SpectrumModel(1,fitted_lines), fitted_losses
		
	end;


	function fit_lines_v0_parallel_experimental(λ_lines::V1, λ::V3, flux::V4, var::V5, T::Type = promote_type(T1,T3,T4,T5); order::Integer ) where
				{ T1<:Number, T3<:Number, T4<:Number, T5<:Number,
				  V1<:AbstractVector{T1}, V3<:AbstractVector{T3}, V4<:AbstractVector{T4}, V5<:AbstractVector{T5}  } 
		
		#fits GH polynomials to an array of absorption lines, taking in an array of λs to fit at and a corresponding σ values for each absorption line. This returns a list of fitted lines (each has the num_gh_orders-ordered GH fit) and a list of losses for each fit, evaluating the fit.
		
		@assert size(λ) == size(flux) == size(var)
		n_pix = length(λ)
		n_lines = length(λ_lines)
		n = length(λ)
		@assert n_lines >= 1 
		@assert 1 <= order <= 10 #10 is the number of gh_polynomials I have computed in the global scope. The code cannot handle more than these many (although we will seldom go beyond 4 orders)
		@assert n_pix > 1 + order*n_lines  # number of fit parameters
		covar = PDiagMat(var)   # diagonal covariance matrix
		design_matrix = ones(n_pix,1)
		
		#fitted_lines = []
		#fitted_losses = []
		
		#fitted_lines = SharedArray{AbsorptionLine}(n_lines, 1)
		#fitted_losses = SharedArray{Float64}(n_lines, 1)
		
		fitted_lines = SharedArray{AbsorptionLine}(n_lines)
		fitted_losses = SharedArray{Float64}(n_lines)
		@sync @distributed for i in 1:n_lines
			λ_line = λ_lines[i]
			σ_line = 0.04
			#will try fitting to every wavelength in the range λ_line-0.5*σ_line to λ_line+0.5*σ_line
			lowest_idx = closest_index(λ, λ_line-fit_devs*σ_line)
			highest_idx = closest_index(λ, λ_line+fit_devs*σ_line)
			
			losses = zeros(highest_idx-lowest_idx+1)
			line_tries = []
			smallest_loss = 999999
			best_fit_idx = -1
			for j = lowest_idx:highest_idx

				line = AbsorptionLine(λ[j], σ_line,(@SVector zeros(order)) ) #create a line data-structure 
				
				design_matrix = hcat(ones(n),		gauss_hermite_basis(line,λ,orders=1:order)  )  	# fit to the line	
				Xt_inv_covar_X = design_matrix' * (covar \ design_matrix) 
				X_inv_covar_y =   design_matrix' * (covar \ flux ) 
				coeff_hat = (Xt_inv_covar_X \ X_inv_covar_y)
				
				
				line = AbsorptionLine(λ[j], σ_line, coeff_hat[2:end] )
				
				
				
				#calculating loss for this fit
				lowest_loss_idx = closest_index(λ, λ_line+(-1*loss_devs*σ_line))
				highest_loss_idx = closest_index(λ, λ_line+loss_devs*σ_line)
				
				loss = 0.0
				#for k = lowest_loss_idx:highest_loss_idx
				#	loss+=abs(flux[k]-line.(λ[k]))
				#end
				
				
				flux_red = flux[lowest_loss_idx:highest_loss_idx]
				
				#var_red = var[lowest_loss_idx:highest_loss_idx]
				#chi_sq = sum(((flux_red.-line.(λ_red)).^2)./var_red)
				line_red = line.(λ[lowest_loss_idx:highest_loss_idx])
				
				@floop ThreadedEx() for k in eachindex(flux_red)
					@reduce(loss += abs(flux_red[k]-line_red[k]))
				end
				
				if loss < smallest_loss
					smallest_loss = loss
					push!(line_tries, line)
					push!(losses, loss)
					best_fit_idx = j
				end
				
				#losses[j-lowest_idx+1] = loss
				
			end
			
			#find fit with lowest loss
			#best_fit_loss = losses[length(losses)]
			
			#best_fit_loss, best_fit_idx = findmin(losses)
			
			best_fit_λ = λ[best_fit_idx]
			#best_fit_line = line_tries[length(line_tries)]
			
			#push!(fitted_lines, best_fit_line)
			#push!(fitted_losses, best_fit_loss)
			fitted_lines[i] = line_tries[length(line_tries)]
			fitted_losses[i] = losses[length(losses)]
		end
		return SpectrumModel(1,fitted_lines), fitted_losses
		
	end;

	function fit_lines_v0_parallel_old(λ_lines::V1, λ::V3, flux::V4, var::V5, T::Type = promote_type(T1,T3,T4,T5); order::Integer ) where
				{ T1<:Number, T3<:Number, T4<:Number, T5<:Number,
				  V1<:AbstractVector{T1}, V3<:AbstractVector{T3}, V4<:AbstractVector{T4}, V5<:AbstractVector{T5}  } 
		
		@assert size(λ) == size(flux) == size(var)
		n_pix = length(λ)
		n_lines = length(λ_lines)
		n = length(λ)
		@assert n_lines >= 1 
		@assert 1 <= order <= 10
		@assert n_pix > 1 + order*n_lines  # number of fit parameters
		covar = PDiagMat(var)   # diagonal covariance matrix
		design_matrix = ones(n_pix,1)
		
		fitted_lines = []
		fitted_losses = []
		Threads.@threads for i in 1:n_lines
			λ_line = λ_lines[i]
			σ_line = 0.04
			#will try fitting to every wavelength in the range λ_line-0.5*σ_line to λ_line+0.5*σ_line
			lowest_idx = closest_index(λ, λ_line-0.25*σ_line)
			highest_idx = closest_index(λ, λ_line+0.25*σ_line)
			
			losses = zeros(highest_idx-lowest_idx+1)
			line_tries = []
			Threads.@threads for j = lowest_idx:highest_idx
				λ_to_fit = λ[j]

				line = AbsorptionLine(λ_to_fit, σ_line,(@SVector zeros(order)) ) #create a line data-structure 
				design_matrix = hcat(ones(n),		gauss_hermite_basis(line,λ,orders=1:order)  )  	# fit to the line	
				Xt_inv_covar_X = design_matrix' * (covar \ design_matrix) 
				X_inv_covar_y =   design_matrix' * (covar \ flux ) 
				coeff_hat = (Xt_inv_covar_X \ X_inv_covar_y)
				line = AbsorptionLine(λ_to_fit, σ_line, coeff_hat[2:end] )
				
				push!(line_tries, line)
				#line_tries.append(line)
				#calculating loss for this fit
				lowest_loss_idx = closest_index(λ, λ_line-2.5*σ_line)
				highest_loss_idx = closest_index(λ, λ_line+2.5*σ_line)
				loss = 0.0
				Threads.@threads for k = lowest_loss_idx:highest_loss_idx
					loss+=abs(flux[k]-line.(λ[k]))
				end
				losses[j-lowest_idx+1] = loss
			end
			
			#find fit with lowest loss
			best_fit_loss, best_fit_idx = findmin(losses)
			best_fit_λ = λ[best_fit_idx]
			best_fit_line = line_tries[best_fit_idx]
			
			push!(fitted_lines, best_fit_line)
			push!(fitted_losses, best_fit_loss)
		end
		return SpectrumModel(1,fitted_lines), fitted_losses
		
	end;



	#Testing 

	function test_fit_perfect(artificial_line::AbsorptionLine, s_or_p::Integer)
				  
		#s_or_p = 0 if it should be a serial test and s_or_p = 1 if a parallel test
		@assert s_or_p == 0 || s_or_p == 1
		#will return \lambda_local, artificial fluxes, fitted0 (this has the lines that were fitted and the loss)
		
		
		
		#creating an artificial line and calculating the fluxes of that line at a local window of lambdas near the wavelength of the artificial line
		#λ_line = λ_lines[i]
			#σ_line = 0.04
		
		
		artificial_λ = artificial_line.λ
		artificial_σ = artificial_line.σ
		
		low_λ = artificial_λ-0.2
		high_λ = artificial_λ+0.2
		points = 1000
		local_λ = ones(points)
		for i in 1:length(local_λ)
			local_λ[i] = low_λ + (high_λ-low_λ)/points * i
		end
		
		x = (local_λ.-artificial_line.λ)./artificial_line.σ
		
		p = mapreduce(i->(artificial_line.gh_coeff[i].*gh_polynomials[i].(x)),hcat,1:length(artificial_line.gh_coeff))
		final_vector = zeros(size(p,1))
		for i in 1:size(p,1)
			local_val = 0.0
			
			for j in 1:size(p, 2)
				local_val = local_val+p[i,j]
			end
			final_vector[i] = local_val
		end
		pp = final_vector
		
		g = exp.(-0.5.*x.^2)
		#ones(λ) .+ sum(g.*p)
		hh = ones(length(local_λ)).+ g.*pp
		
		
		#fitting to the artificial line
		
		λ_lines = [artificial_λ]
		vars = ones(length(hh)).*artificial_σ
		if s_or_p == 0
		
			fitted0 = fit_lines_v0_serial(λ_lines,local_λ,hh, vars, order = 4)
		elseif s_or_p == 1
			#fitted0 = fit_lines_v0_parallel(λ_lines,local_λ,hh, vars, order = 4)
			fitted0 = fit_lines_v0_parallel_experimental(λ_lines,local_λ,hh, vars, order = 4)
		end
		
		

		return [local_λ, hh, fitted0 ]

		end;





	function test_fit_delta(wavelength::T1) where{T1<:Number}
				  
		#will return \lambda_local, artificial fluxes, fitted0 (this has the lines that were fitted and the loss)
		
		
		
		#creating an artificial line and calculating the fluxes of that line at a local window of lambdas near the wavelength of the artificial line
		#λ_line = λ_lines[i]
			#σ_line = 0.04
		
		artificial_λ = wavelength
		artificial_σ = 0.04
		
		low_λ = artificial_λ-0.2
		high_λ = artificial_λ+0.2
		points = 1000
		local_λ = ones(points)
		hh = ones(points)
		left_dip = points * rand()/2
		right_dip = points/(rand()*2)
		for i in 1:length(local_λ)
			local_λ[i] = low_λ + (high_λ-low_λ)/points * i

			if (i > left_dip && i < right_dip)
				hh[i] = 0.0
			end
		end
		
		
		
		
		#fitting to the artificial line
		
		λ_lines = [artificial_λ]
		vars = ones(length(hh)).*artificial_σ
		
		fitted_serial = fit_lines_v0_serial(λ_lines,local_λ,hh, vars, order = 4)
		fitted_parallel = fit_lines_v0_parallel_experimental(λ_lines,local_λ,hh, vars, order = 4)
		
		
		
		

		return [local_λ, hh, fitted_serial, fitted_parallel ]
		end;
		
end