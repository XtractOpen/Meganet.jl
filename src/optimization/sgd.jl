export SGD, solve

"""
Stochastic Gradient Descent
"""
type SGD
    maxEpochs::Int
    miniBatch::Int
    out::Bool
    learningRate
    momentum::Real
    nesterov::Bool
	ADAM::Bool
    function SGD(;maxEpochs=10,miniBatch=16,out=true,learningRate=0.1,momentum=0.9,nesterov=false,ADAM=false)
        if ADAM && nesterov
            warn("sgd(): ADAM and nestrov together - choosing ADAM");
            nesterov  = false;
        end
        new(maxEpochs,miniBatch,out,learningRate,momentum, nesterov, ADAM)
    end
end

Base.display(this::SGD)=println("SGD(maxEpochs=$(this.maxEpochs),miniBatch=$(this.miniBatch),learningRate=$(this.learningRate),momentum=$(this.momentum),nesterov=$(this.nesterov),ADAM=$(this.ADAM))")

function solve(this,objFun::dnnObjFctn,xc,Y,C,Yv,Cv)

    # evaluate training and validation
    epoch = 1;
    xOld = copy(xc);
    dJ = 0*xc
    mJ = 0*xc
    vJ = 0*xc
    if this.ADAM
        mJ = 0*xc
        vJ = 0*xc
    end
    beta2 = 0.999;
    beta1 = this.momentum;

    lr    = this.learningRate

    if this.out; display(this); end


    while epoch <= this.maxEpochs
        nex = size(Y,2)
        ids = randperm(nex)

        for k=1:Int(ceil(nex/this.miniBatch))
            idk = ids[(k-1)*this.miniBatch+1: min(k*this.miniBatch,nex)]
            if this.nesterov && !this.ADAM
                Jk,dummy,dJk = evalObjFctn(objFun,xc-this.momentum*dJ,Y[:,idk],C[:,idk]);
            else
                Jk,dummy,dJk = evalObjFctn(objFun,xc,Y[:,idk],C[:,idk]);
            end

            if this.ADAM
               mJ = beta1*mJ + (1-beta1)*(dJk)
               vJ = beta2*vJ + (1-beta2)*(dJk.^2)
#              dJ = lr*((mJ./(1-beta1^(epoch)))./sqrt.((vJ./(1-beta2^(epoch)))+1e-8))
            else
               dJ = lr*dJk + this.momentum*dJ
            end
            xc = xc - dJ
        end
        # we sample 2^12 images from the training set for displaying the objective.
        idt     = ids[1:min(nex,2^12)]
        Jc,para   = evalObjFctn(objFun,xc,Y[:,idt],C[:,idt]);
        Jval,pVal = getMisfit(objFun,xc,Yv,Cv,false);

        if this.out;
            @printf "%d\t%1.2e\t%1.2f\t%1.2e\t%1.2e\t%1.2f\n" epoch Jc 100*(1-para[3]/para[2]) norm(xOld-xc) Jval 100*(1-pVal[3]/para[2])
        end

        xOld       = copy(xc);
        epoch = epoch + 1;
    end
    # His = struct('str',{str},'frmt',{frmt},'his',his(1:min(epoch,this.maxEpochs),:));
    return xc
end
