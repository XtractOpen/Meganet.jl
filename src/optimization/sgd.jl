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

    global XC
    XC = xc

    # evaluate training and validation
    epoch = 1
    Jtrain_old = Inf
    xOld = copy(xc)
    dJ = zeros(T,size(xc))
    mJ = zeros(T,size(xc))
    vJ = zeros(T,size(xc))
    beta2 = convert(T,0.999)
    beta1 = this.momentum
    lr    = this.learningRate

    this.out && display(this)

    # Distribute the data
    nw = nworkers()
    Ys = SharedArray(Y)
    Cs = SharedArray(C)
    println("Using $(nw) workers...")
    tmp = Array{Any}(0,0)
    dJ = Vector{Vector{T}}(nw)
    TEST = Vector{T}(nw)

    while epoch <= this.maxEpochs

        # Shuffle and balance
        nex = size(Ys,2)
        ids = randperm(nex)

        for k=1:ceil(Int64,nex/this.miniBatch)
            idk = ids[(k-1)*this.miniBatch+1: min(k*this.miniBatch,nex)]
            indices = balance(length(idk), 1, 2)

            println("""
            idk :   $(idk)
            ind1 :   $(idk[indices[1]])
            ind2 :   $(idk[indices[2]])
            """)
            J, test = train(this, objFun, xc, Ys, Cs, beta1, beta2, idk)
            J1, test1 = train(this, objFun, xc, Ys, Cs, beta1, beta2, idk[indices[1]])
            J2, test2 = train(this, objFun, xc, Ys, Cs, beta1, beta2, idk[indices[2]])
            A1,B1 = splitWeights(objFun, test)
            A2,B2 = splitWeights(objFun, test1 .+ test2)
            println("Theta          : ", norm(A1 - A2))
            println("Weights        : ", norm(B1 - B2))
            println("""
            J               : $J
            avg             : $((J2 + J1)/2)

            Residual        : $(norm(test .- (test1 .+ test2)))
            %               : $(100*norm(test .- (test1+ test2))/ norm(test))
            """)
            for (i, pid) in enumerate(Ys.pids)
                TEST[i], dJ[i] = @fetchfrom pid train(this, objFun, xc, Ys, Cs, beta1, beta2, idk[indices[i]])
            end

            #=Plots.pyplot()
            p = Plots.plot(test)
            Plots.plot!(p, test1 .+ test2)
            Plots.display(p)
            sleep(100) =#

            #println("Residual ($(nw) workers) : ", norm(test - sum(dJ)./2), "  |  ", 100*(norm(test-sum(dJ)./2)/norm(test)), "%")
            println("""
            C1 Residual :   $(norm(test1 - dJ[1]))
            C2 Residual :   $(norm(test2 - dJ[2]))

            S Residual  :   $(norm((test1+test2)./2 - test))
            """)
            xc .= xc .- sum(dJ)
        end

        # we sample 2^12 images from the training set for displaying the objective.
        idt     = ids[1:min(nex,2^12)]
        Jtrain,ptrain   = getMisfit(objFun,xc,Y[:,idt],C[:,idt],tmp,false);
        Jval,pVal = getMisfit(objFun,xc,Yv,Cv,tmp,false);

        if this.out;
            s = @sprintf "%d\t%1.2e\t%1.2f\t%1.2e\t%1.2e\t%1.2f\n" epoch Jtrain 100*(1-ptrain[3]/ptrain[2]) norm(xOld-xc) Jval 100*(1-pVal[3]/pVal[2])
            Jtrain < Jtrain_old ? (color = :light_green) : (color = :light_red)
            print_with_color(color, s)
            Jtrain_old = Jtrain
        end

        xOld       = copy(xc);
        epoch = epoch + 1;
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

    return xc
end

"""
    Use: evalObjFctn_local(objFun, xc, Y, C, n)

Evaluate the objective function on `n` random examples from `Y`
"""
function evalObjFctn_local(objFun::dnnObjFctn, xc::Array{T,1}, Y::SharedArray{T,2}, C::SharedArray{T,2}, n::Int, ids::Vector{<:Integer}) where {T<:Number}

    nex = length(ids)
    idt = ids[1:min(n, nex)]
    tmp = Array{Any}(0,0)

    Jc, hisF, dJ = evalObjFctn(objFun,xc,Y[:,idt], C[:,idt], tmp, false);

    return Jc, hisF, dJ
end

"""
Train on the local part of the distributed data in Y
"""
function train(this::SGD{T}, objFun::dnnObjFctn, xc::Array{T,1}, Y::SharedArray{T,2}, C::SharedArray{T,2}, beta1::T, beta2::T, ids::Vector{<:Integer}) where {T<:Number}
# TODO send the worker SGD and objFun onl once

    #println("Examples : ", length(ids))

    idk = ids
    nex = length(ids)
    nworkers = length(Y.pids)
    lr = this.learningRate
    #lr = this.learningRate*nworkers
    dJ = zeros(T,size(xc))
    tmp = Array{Any}(0,0)

    #for k=1:ceil(Int64,nex/this.miniBatch)
        if this.nesterov && !this.ADAM
            Jk,dummy,dJk = evalObjFctn(objFun, copy(xc)-this.momentum*dJ, copy(Y[:,idk]), copy(C[:,idk]), copy(tmp));
        else
            Jk,dummy,dJk = evalObjFctn(objFun, copy(xc), copy(Y[:,idk]), copy(C[:,idk]), copy(tmp));
        end

        if this.ADAM
           mJ = beta1*mJ + (one(T)-beta1)*(dJk)
           vJ = beta2*vJ + (one(T)-beta2)*(dJk.^2)
        else
           dJ = lr*dJk + this.momentum*dJ
        end

        # Exchange weights
        #update = Future(1)
        #put!(update, dJ)
        #xc = @fetchfrom 1 update_weights!(Meganet.XC, update)
    #end

    return Jk, dJ
end

function update_weights!(xc::Vector{<:Number}, update::Future)
    println("Derivative received from :$(update.whence)")
    dJ = fetch(update)
    xc .= xc .- dJ

    return xc
end
