"Package demonstrating how to make your code into a Julia package"
module ExamplePkg

using Distributions

export pdf_normal

"""
   pdf_normal(x, mu, sigma)

Return pdf of a normal distribution with mean mu and standard deviation sigma at point x.
"""
function pdf_normal(x::Number, mu::Number, sigma::Number)
  dist = Normal(mu,sigma)
  pdf(dist,x)
end

end # module