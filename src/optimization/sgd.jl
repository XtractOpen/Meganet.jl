export SGD, solve, getSGDsolver, solve_local

"""
Stochastic Gradient Descent
"""
mutable struct SGD{T}
    maxEpochs::Int
    miniBatch::Int
    out::Bool
    learningRate::T
    momentum::T
    nesterov::Bool
	ADAM::Bool
end

function getSGDsolver(TYPE::Type ;maxEpochs=10,miniBatch=16,out=true,learningRate=0.1,momentum=0.9,nesterov=false,ADAM=false)
	if ADAM && nesterov
		warn("sgd(): ADAM and nestrov together - choosing ADAM");
		nesterov  = false;
	end
	return SGD{TYPE}(maxEpochs,miniBatch,out,convert(TYPE,learningRate),convert(TYPE,momentum), nesterov, ADAM)
end



Base.display(this::SGD)=println("SGD(maxEpochs=$(this.maxEpochs),miniBatch=$(this.miniBatch),learningRate=$(this.learningRate),momentum=$(this.momentum),nesterov=$(this.nesterov),ADAM=$(this.ADAM))")

function solve(this::SGD{T},objFun::dnnObjFctn,xc::Array{T},Y::Array{T},C::Array{T},Yv::Array{T},Cv::Array{T}) where {T}

    global XC
    XC = xc

    # evaluate training and validation
    epoch = 1
    xOld = copy(xc)
    dJ = zeros(T,size(xc))
    mJ = zeros(T,size(xc))
    vJ = zeros(T,size(xc))
    #if this.ADAM              ## redundant
    #    mJ = zeros(T,size(xc));
    #    vJ = zeros(T,size(xc));
    #end
    beta2 = convert(T,0.999)
    beta1 = this.momentum
    lr    = this.learningRate

    this.out && display(this)

    # Distribute the data
    nw = nworkers()
    Yd = distribute(Y, dist = (1, nw))
    Cd = distribute(C, dist = (1, nw))
    println("Using $(nw) workers...")


    while epoch <= this.maxEpochs

        # Train on all workers
        println("--------------------------------------------")
        for pid in Yd.pids
        #@sync for pid in Yd.pids
            @fetchfrom pid train(this, objFun, xc, Yd, Cd, beta1, beta2)
            #@async @fetchfrom pid train(this, objFun, xc, Yd, Cd, beta1, beta2)
        end
        println("--------------------------------------------")

        # we sample 2^12 images from the training set for displaying the objective.
        xc = Meganet.XC
        nex = size(Y,2)
        n_total = min(nex,2^12)
        n_worker = div(n_total, nw)

        JcA     = Array{T,1}(nw)
        hisFA    = Array{Array{T,1},1}(nw)
        dJA      = Array{Array{T,1},1}(nw)

        for (i, pid) in enumerate(Yd.pids)
            JcA[i], hisFA[i], dJA[i] = evalObjFctn_local(objFun, xc, Yd, Cd, n_worker)
        end

        Jc = sum(JcA)
        hisF = sum(hisFA)
        dJ = sum(dJA)

        Jval,pVal   = getMisfit(objFun,xc,Yv,Cv,false);

        if this.out;
            @printf "%d\t%1.2e\t%1.2f\t%1.2e\t%1.2e\t%1.2f\n" epoch Jc 100*(1-dJ/hisF) norm(xOld-xc) Jval 100*(1-pVal[3]/pVal[2])
        end

        xOld   = copy(xc);
        epoch += 1;
    end

    return xc
end

"""
    Use: evalObjFctn_local(objFun, xc, Y, C, n)

Evaluate the objective function on `n` random examples from `Y`
"""
function evalObjFctn_local(objFun::dnnObjFctn, xc::Array{T,1}, Y::DArray{T,2}, C::DArray{T,2}, n::Int) where {T<:Number}

    Y_local = localpart(Y)
    C_local = localpart(C)

    nex = size(Y_local,2)
    ids = randperm(nex)
    idt = ids[1:n]

    Jc, hisF, dJ = evalObjFctn(objFun,xc,Y_local[:,idt], C_local[:,idt]);

    return Jc, hisF, dJ

end

"""
Train on the local part of the distributed data in Y
"""
function train(this::SGD{T}, objFun::dnnObjFctn, xc::Array{T,1}, Y::DArray{T,2}, C::DArray{T,2}, beta1::T, beta2::T) where {T<:Number}
# TODO send the worker SGD and objFun onl once

    Y_local = localpart(Y)
    C_local = localpart(C)

    nex = size(Y_local,2)
    ids = randperm(nex)
    lr = this.learningRate
    dJ = zeros(T,size(xc))

    for k=1:ceil(Int64,nex/this.miniBatch)
        println("k  : $k")
        idk = ids[(k-1)*this.miniBatch+1: min(k*this.miniBatch,nex)]
        if this.nesterov && !this.ADAM
            Jk,dummy,dJk = evalObjFctn(objFun, xc-this.momentum*dJ, Y_local[:,idk], C_local[:,idk]);
        else
            Jk,dummy,dJk = evalObjFctn(objFun, xc, Y_local[:,idk], C_local[:,idk]);
        end

        if this.ADAM
           mJ = beta1*mJ + (one(T)-beta1)*(dJk)
           vJ = beta2*vJ + (one(T)-beta2)*(dJk.^2)
        else
           dJ = lr*dJk + this.momentum*dJ
        end
        #xc = xc - dJ

        # Exchange weights
        update = Future(1)
        put!(update, dJ)
        xc = @fetchfrom 1 update_weights!(Meganet.XC, update)
    end

    return nothing
end

function update_weights!(xc::Vector{<:Number}, update::Future)
    dJ = fetch(update)
    xc .= xc .- dJ

    return xc
end
