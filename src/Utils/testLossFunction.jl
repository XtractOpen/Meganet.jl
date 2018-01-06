export testLossFunction

function testLossFunction(S;nf=10,nex=20,nc=7,out=false)

Y   = randn(nf,nex)/10
nw2 = (S.addBias) ? nf+1 : nf
W   = randn(nc,nw2)/10
C   = full(sparse(full(rand(1:nc,nex)),collect(1:nex),ones(nex),nc,nex))

@testset "test dW" begin
function testdWFun(x,v=[])
    x = reshape(x,nc,nw2)
    if !(isempty(v))
        F, para, dWF, = getMisfit(S,x,Y,C,false,true)
        return F, dot(dWF,v)
    else
        return getMisfit(S,x,Y,C,false,false)[1]
    end
end
chkDer, = checkDerivative(testdWFun,copy(vec(W)),out=out)
@test chkDer
end

@testset "test d2W" begin
function testd2WFun(x,v=[])
    x = reshape(x,nc,nw2)
    if !(isempty(v))
        F, para, dWF,d2F = getMisfit(S,x,Y,C,false,true)
        return vec(dWF),d2F*vec(v)
    else
        return vec(getMisfit(S,x,Y,C,false,true)[3])
    end
end
chkDer, = checkDerivative(testd2WFun,copy(vec(W)),out=out)
@test chkDer
end

@testset "test dY" begin
function testdYFun(x,v=[])
    x = reshape(x,nf,nex)
    if !(isempty(v))
        F, para, dWF,d2WF,dYF = getMisfit(S,W,x,C,true,false)
        return F, dot(dYF,v)
    else
        return getMisfit(S,W,x,C,false,false)[1]
    end
end
chkDer, = checkDerivative(testdYFun,copy(vec(Y)),out=out)
@test chkDer
end

@testset "test d2Y" begin
function testd2YFun(x,v=[])
    x = reshape(x,nf,nex)
    if !(isempty(v))
        F, para, dWF,d2WF,dYF,d2YF = getMisfit(S,W,x,C,true,false)
        return vec(dYF),d2YF*vec(v)
    else
        return vec(getMisfit(S,W,x,C,true,false)[5])
    end
end
chkDer, = checkDerivative(testd2YFun,copy(vec(Y)),out=out)
@test chkDer
end
end
