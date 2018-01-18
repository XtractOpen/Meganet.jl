export TikhonovReg, getTikhonovReg, regularizer

"""
Tikhnov Regularizer R(x) = 0.5*alpha*(H*(x-xref), x-xref), with H spd
"""
type TikhonovReg{T}
    H
    alpha::T
    xref::Array{T}
end

getTikhonovReg(TYPE;H=I,alpha=1.0,xref=zeros(0))=TikhonovReg{TYPE}(H,convert(TYPE,alpha),convert.(TYPE,vec(xref)))

function regularizer(this::TikhonovReg,x)
    u = vec(copy(x))
    if !isempty(this.xref)
        u.-=this.xref
    end
    d2S = this.alpha*this.H
    dS = d2S*u
    S  = 0.5*dot(dS,u)
    return S,dS,d2S
end
