export checkDerivative, getRandomTestDirection

function getRandomTestDirection(x0::Float64)
	return randn()
end
function getRandomTestDirection(x0::Complex128)
	return randn()+1im*randn()
end
function getRandomTestDirection(x0::Array{T}) where {T<:AbstractFloat}
	return randn(T,size(x0))
end
function getRandomTestDirection(x0::Array{Complex128})
	return randn(size(x0))+1im*randn(size(x0))
end


function checkDerivative(f::Function,df::Function,x0;kwargs...)
	function testFun(x,v=[])
		if !(isempty(v))
			return f(x), df(x)*v
		else
			return f(x)
		end
	end
	return checkDerivative(testFun,x0;kwargs...)
end


function checkDerivative(f::Function,x0::Array{T};out::Bool=true,tol::Float64=1.9,nSuccess::Int=4,v=getRandomTestDirection(x0),nTrials=20) where T
# checkDerivative(f::Function,x0;out::Bool=true,tol::Float64=1.9,nSuccess::Int=3)
	if out
		println(@sprintf("%9s\t%9s\t%9s\t%9s\t%9s\t%5s","h","E0","E1","O1","O2","OK?"))
	end

	f0,dvf  = f(x0,v)
	nf0     = norm(f0)
	#dvf    = real(dvf)
	Error   = zeros(nTrials,2)
	Order   = zeros(nTrials,2)
	Success = zeros(nTrials)
	for j=1:nTrials
		hj = convert(T,2.0^(-j))
		ft = f(x0+hj*v)            # function value
		Error[j,1] = norm(f0-ft)/nf0          # Error TaylorPoly 0
		Error[j,2] = norm(f0 .+hj*dvf .- ft)/nf0 # Error TaylorPoly 1
		if j>1
			Order[j,:] = log2.(Error[j-1,:]./Error[j,:]);
		end
		if (Order[j,2]>tol) || (Error[j,1]/Error[j,2] > 100); Success[j]=1; end
		if out
			println(@sprintf("%1.3e\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t%5d",
							hj, Error[j,1],Error[j,2], Order[j,1],Order[j,2],Success[j]))
		end
	end
	pass = sum(Success) > nSuccess
	return  pass,Error,Order
end
