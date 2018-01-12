export convFFTKernel, getEigs
<<<<<<< HEAD

type convFFTKernel
    nImg  # image size
=======
## For the functions nImgIn, nImgOut, nFeatIn, nFeatOut, nTheta, getOp, initTheta : see abstractConvKernel.jl
## All convKernel types are assumed to have fields nImage and sK
type convFFTKernel <: abstractConvKernel
    nImg
>>>>>>> master
    sK
    S
    function convFFTKernel(nImg,sK)
        S = getEigs(nImg,sK)
        return new(nImg,sK,S)
    end
end

function getEigs(nImg,sK)
    S = zeros(Complex128,prod(nImg),prod(sK[1:2]));
    for k=1:prod(sK[1:2])
        Kk = zeros(sK[1],sK[2]);
        Kk[k] = 1;
        Ak = getConvMatPeriodic(Kk,[nImg[1],nImg[2], 1]);

        S[:,k] = vec(fft2( reshape(full(Ak[:,1]),nImg[1],nImg[2]) ));
    end
    return S
end


function Amv(this::convFFTKernel,theta,Y)

    nex   = div(numel(Y),prod(nImgIn(this)))

    # compute convolution
    AY    = zeros(Complex128,tuple([nImgOut(this);  nex]...));
    theta = reshape(theta, tuple([prod(this.sK[1:2]); this.sK[3:4]]...));
    Yh    = ifft2(reshape(Y,tuple([nImgIn(this); nex]...)));

    #### allocate stuff for the loop
    Sk = zeros(Complex128,tuple(nImgOut(this)...))
    T  = zeros(Complex128,tuple(nImgOut(this)...))
    nn = nImgOut(this); nn[3] = 1;
    sumT = zeros(Complex128,tuple([nn;nex]...))
    ####
       
    for k=1:this.sK[4]
        Sk = reshape(this.S*theta[:,:,k],tuple(nImgIn(this)...));
        #T  = Sk .* Yh;
        #sumT = sum(T,3)
        sumT = hadamardSum(sumT,Yh,Sk)
        AY[:,:,k,:]  = sumT[:,:,1,:];
    end
    AY = real(fft2(AY));
    Y  = reshape(AY,:,nex);
    return Y
end

function ATmv(this::convFFTKernel,theta,Z)
    
    nex   =  div(numel(Z),prod(nImgOut(this)));
    ATY   = zeros(Complex128,tuple([nImgIn(this); nex]...));
    theta = reshape(theta, prod(this.sK[1:2]),this.sK[3],this.sK[4]);
    
    Yh = fft2(reshape(Z,tuple([nImgOut(this); nex]...)));
    for k=1:this.sK[3]
        tk = theta[:,k,:]
        #if size(this.S,2) == 1
        #    tk = reshape(tk,1,:);
        #end
        Sk = reshape(this.S*tk,tuple(nImgOut(this)...));
        #T  = Sk.*Yh;
        #sumT = sum(T,3)
        sumT = hadamardSum(sumT,Yh,Sk)
        ATY[:,:,k] = sumT[:,:,1];
    end
    ATY = real(ifft2(ATY));
    ATY = reshape(ATY,:,nex);
    return ATY
end

function Jthetamv(this::convFFTKernel,dtheta,dummy,Y,temp=nothing)

    nex    =  div(numel(Y),nFeatIn(this));
    Y      = reshape(Y,:,nex);
    Z      = Amv(this,dtheta,Y);
    return Z
end

function JthetaTmv(this::convFFTKernel,Z,dummy,Y)
    #  derivative of Z*(A(theta)*Y) w.r.t. theta

    nex  =  div(numel(Y),nFeatIn(this));

    dth1  = zeros(this.sK[1]*this.sK[2],this.sK[3],this.sK[4]);
    Y     = permutedims(reshape(Y,tuple([nImgIn(this); nex ]...)),[1 2 4 3]);
    Yh    = reshape(fft2(Y),prod(this.nImg[1:2]),nex*this.sK[3]);
    Zh    = permutedims(ifft2(reshape(Z,tuple([nImgOut(this); nex]...))),[1 2 4 3]);
    Zh     = reshape(Zh,:, this.sK[4]);

    for k=1:prod(this.sK[1:2])
        temp = conj(this.S[:,k]) .* Yh
        temp = reshape(temp,:,this.sK[3])
        dth1[k,:,:] = real(conj(temp)'*Zh);
    end

    dtheta = reshape(dth1,tuple(this.sK...));
    return dtheta
end
<<<<<<< HEAD


function getOp(this::convFFTKernel,theta)

    m = prod(nImgOut(this))
    n = prod(nImgIn(this))

    A = LinearOperator(m,n,false,false,
                        v -> Amv(this,theta,v),
                        Nullable{Function}(),
                        w -> ATmv(this,theta,w))
    return A
end

function initTheta(this::convFFTKernel)

    sd    = 0.1;
    theta = sd*randn(prod(this.sK));
    #id1 = find(theta>2*sd);
    #theta(id1[:]) = randn(numel(id1),1);

    #id2 = find(theta< -2*sd);
    #theta(id2(:)) = randn(numel(id2),1);
    #theta = max(min(2*sd, theta),-2*sd);
    return theta
end

function nFeatIn(this::convFFTKernel)
    return prod(nImgIn(this));
end
function nFeatOut(this::convFFTKernel)
    return prod(nImgOut(this));
end

function nTheta(this)
    return prod(this.sK);
end

function hadamardSum(sumT::Array{Complex128},Yh::Array{Complex128},Sk::Array{Complex128})
    sumT .= Complex(0.0)
    for i4 = 1:size(Yh,4)
        for i3 = 1:size(Yh,3)
            for i2 = 1:size(Yh,2)
                for i1 = 1:size(Yh,1)
                    @inbounds tt = Sk[i1,i2,i3]
                    @inbounds sumT[i1,i2,1,i4] += tt * Yh[i1,i2,i3,i4]
                end
            end
        end
    end        
    return sumT
end

=======
>>>>>>> master
