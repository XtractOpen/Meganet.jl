export SGD, solve, getSGDsolver, solve_local
#import Plots

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

    #global XC
    #XC = xc

    # evaluate training and validation
    epoch = 1
    Jtrain_old = Inf
    xOld = copy(xc)
    mJ = zeros(T,size(xc))
    vJ = zeros(T,size(xc))
    beta2 = convert(T,0.999)
    beta1 = this.momentum
    lr    = this.learningRate
    history_length = floor(Int, size(Y,2)/this.miniBatch)
    this.out && display(this)

    # Distribute the data
    nw = nworkers()
    Ys = SharedArray(Y)
    Cs = SharedArray(C)
    Yvs = SharedArray(Yv)
    Cvs = SharedArray(Cv)
    println("Using $(nw) workers...")

    # Pre-allocate output
    tmp         = Array{Any}(0,0)
    dJA        = Vector{Vector{T}}(nw)
    hisFA       = Vector{T}(nw)
    JcA         = Vector{T}(nw)
    dJ          = zeros(T, length(xc))
    his         = Vector{T}(history_length)
    Jtrain      = Vector{T}(history_length); count = 1

    # Initialize training validation history
    inf = convert(T, Inf)
    fill!(his, inf)
    fill!(Jtrain, inf)

    while epoch <= this.maxEpochs

        # Shuffle and balance
        nex = size(Ys,2)
        ids = randperm(nex)

        @time for k=1:ceil(Int64,nex/(this.miniBatch))
            idk = ids[(k-1)*this.miniBatch+1: min(k*this.miniBatch,nex)]
            indices = balance(length(idk), 1, nw)

            # Distribute batch across workers
            @sync for (i, pid) in enumerate(Ys.pids)
                if this.nesterov && !this.ADAM
                    @async JcA[i], hisFA[i], dJA[i] = @fetchfrom pid train(this, objFun, xc .- this.momentum.*dJ, Ys, Cs, beta1, beta2, idk[indices[i]])
                    #JcA[i], hisFA[i], dJA[i] = train(this, objFun, xc .- this.momentum.*dJ, Ys, Cs, beta1, beta2, idk[indices[i]])
                else
                    @async JcA[i], hisFA[i], dJA[i] = @fetchfrom pid train(this, objFun, xc, Ys, Cs, beta1, beta2, idk[indices[i]])
                    #JcA[i], hisFA[i], dJA[i] = train(this, objFun, xc, Ys, Cs, beta1, beta2, idk[indices[i]])
                end
            end

            # Average results from workers
            Jc       = sum(JcA)./nw
            hisF     = sum(hisFA)./nw
            dJk      = sum(dJA)./nw

            # Update training validation history
            his[count%history_length + 1] = hisF
            Jtrain[count%history_length + 1] = Jc
            count += 1

            # Update weights
            if this.ADAM
               mJ = beta1*mJ + (one(T)-beta1)*(dJk)
               vJ = beta2*vJ + (one(T)-beta2)*(dJk.^2)
            else
               dJ .= lr .* dJk .+ this.momentum .* dJ
            end
            xc .= xc .- dJ
        end

        # Validate, and print status
        @time begin
        nex = size(Yv,2)
        indices = balance(nex, 1, nw)
        JvalA = Vector{T}(nw)
        PvalA = Vector{T}(nw)
        @sync for (i, pid) in enumerate(Yvs.pids)
            @async JvalA[i], PvalA[i] = @fetchfrom pid evalObjFctn_local(objFun, xc, Yvs, Cvs, collect(indices[i]));
        end
        Jval = sum(JvalA)./nw
        Pval = sum(PvalA)./nw
        end

        if this.out
            Jhat = mean(Jtrain)
            s = @sprintf "%d\t%1.2e\t%1.2f\t%1.2e\t%1.2e\t%1.2f\n" epoch Jhat 100*(1-mean(his)) norm(xOld-xc) Jval 100*(1-Pval)
            Jhat < Jtrain_old ? (color = :light_green) : (color = :light_red)
            print_with_color(color, s)
            Jtrain_old = Jhat
        end

        xOld   = copy(xc);
        epoch += 1;
    end
    return xc
end

"""
Train on the local part of the shared data in Y
"""
function train(this::SGD{T}, objFun::dnnObjFctn, xc::Array{T,1}, Y::SharedArray{T,2}, C::SharedArray{T,2}, beta1::T, beta2::T, idk::Vector{<:Integer}) where {T<:Number}

    nex = length(idk)
    nworkers = length(Y.pids)
    lr = this.learningRate
    #lr = this.learningRate*nworkers
    dJ = zeros(T,size(xc))
    tmp = Array{Any}(0,0)

    # compute misfit
    theta,W = splitWeights(objFun, xc)
    Fc,hisF,dFth,dFW,tmp = getMisfit(objFun, theta, W, Y[:, idk], C[:, idk], tmp, true)

    # Regularize
    Rth,dRth = regularizer(objFun.pRegTheta, theta)
    RW,dRW = regularizer(objFun.pRegW, W)
    Jc = Fc + Rth + RW
    ntheta = nTheta(objFun.net)
    dJ[1:ntheta]       .= dFth .+ dRth
    dJ[(ntheta+1):end] .= dFW  .+ dRW
    his = hisF[3]/hisF[2]

    return Jc, his, dJ
end

"""
    Use: evalObjFctn_local(objFun, xc, Y, C, n)

Evaluate the objective function on `n` random examples from `Y`
"""
function evalObjFctn_local(objFun::dnnObjFctn, xc::Array{T,1}, Y::SharedArray{T,2}, C::SharedArray{T,2}, idk::Vector{<:Integer}) where {T<:Number}

    tmp = Array{Any}(0,0)
    Jc, hisF, dJ = evalObjFctn(objFun,xc,Y[:,idk], C[:,idk], tmp, false);

    return Jc, hisF[3]/hisF[2]
end


function update_weights!(xc::Vector{<:Number}, update::Future)
    println("Derivative received from :$(update.whence)")
    dJ = fetch(update)
    xc .= xc .- dJ

    return xc
end

    #= while epoch <= this.maxEpochs
        tic()

        # Shuffle and balance
        nex = size(Y,2)
        ids = randperm(nex)
        indices = balance(nex, this.miniBatch, nw)

        # Train on all workers
        #@sync for pid in Ys.pids
        #    @async @fetchfrom pid train(this, objFun, xc, Ys, Cs, beta1, beta2)
        #end

        for (i, pid) in enumerate(Ys.pids)
            @fetchfrom pid train(this, objFun, xc, Ys, Cs, beta1, beta2, ids[indices[i]])
        end

        # we sample 2^12 images from the training set for displaying the objective.
        xc = Meganet.XC
        nex = size(Y,2)
        n_total = min(nex,60)
        n_worker = div(n_total, nw)

        JcA     = Array{T,1}(nw)
        hisFA    = Array{Array{T,1},1}(nw)
        Jval = zero(T)
        pVal = Array{T,1}()
	    tmp = Array{Any}(0,0)

        @sync begin
                for (i, pid) in enumerate(Ys.pids)
                    @async JcA[i], hisFA[i] = @fetchfrom pid evalObjFctn_local(objFun, xc, Ys, Cs, n_worker, ids[indices[i]])
                end

                # Currently validation data is on master so this is local
                @async Jval,pVal = @fetchfrom 1 getMisfit(objFun,xc,Yv,Cv,tmp, false);
        end

        Jc = sum(JcA)
        hisF = sum(hisFA)

        if this.out;
            s = @sprintf "%d\t%1.2e\t%1.2f\t%1.2e\t%1.2e\t%1.2f\n" epoch Jc 100*(1-hisF[3]/hisF[2]) norm(xOld-xc) Jval 100*(1-pVal[3]/pVal[2])
            Jc < Jc_old ? (color = :light_green) : (color = :light_red)
            print_with_color(color, s)
            Jc_old = Jc
        end

        xOld   = copy(xc);
        epoch += 1
        toc()
    end =#
