export TikhonovReg, regularizer

"""
Tikhnov Regularizer R(x) = 0.5*alpha*(H*(x-xref), x-xref), with H spd
"""
type TikhonovReg
    H
    alpha::Real
    xref
    TikhonovReg(;H=I,alpha=1.0,xref=zeros(0))=new(H,alpha,vec(xref))
end

function regularizer(this::TikhonovReg,x)
    u = vec(copy(x))
    if !isempty(this.xref)
        u.-=xref
    end
    d2S = this.alpha*this.H
    dS = d2S*u
    S  = 0.5*dot(dS,u)
    return S,dS,d2S
end
