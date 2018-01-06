export testAbstractMeganetElement

"""
simple tests that every AbstractMeganetElement must pass
"""
function testAbstractMeganetElement(L;out::Bool=false,nex=8)

    @testset "features immutable" begin
        theta = initTheta(L)
        Y     = randn(nFeatIn(L),nex)
        Yo    = copy(Y)
        Zd,Z,tmp  = apply(L,theta,Y,true)
        @test norm(Y-Yo)/norm(Yo) < 1e-10

        dY    = randn(nFeatIn(L),nex)
        Z1    =  JYmv(L,dY,theta,Y,tmp)[2]
        @test norm(Y-Yo)/norm(Yo) < 1e-10
        Z12    =  JYmv(L,dY,theta,Y,tmp)[2]
        @test norm(Z1-Z12)/(norm(Z1)+(norm(Z1)==0)) < 1e-10

        dth    = randn(nTheta(L))
        Z1    =  Jthetamv(L,dth,theta,Y,tmp)[2]
        @test norm(Y-Yo)/norm(Yo) < 1e-10
        Z12    =  Jthetamv(L,dth,theta,Y,tmp)[2]
        @test norm(Z1-Z12)/(norm(Z1)+(norm(Z1)==0)) < 1e-10

        Zout = randn(nFeatOut(L),nex)
        Z1 =  JthetaTmv(L,copy(Zout),[],theta,Y,tmp)
        @test norm(Y-Yo)/norm(Yo) < 1e-10
        Z12 =  JthetaTmv(L,copy(Zout),[],theta,Y,tmp)
        @test norm(Z1-Z12)/(norm(Z1)+(norm(Z1)==0)) < 1e-10

        Zout = randn(nFeatOut(L),nex)
        Z1 =  JYTmv(L,copy(Zout),[],theta,Y,tmp)
        @test norm(Y-Yo)/norm(Yo) < 1e-10
        Z12 =  JYTmv(L,copy(Zout),[],theta,Y,tmp)
        @test norm(Z1-Z12)/(norm(Z1)+(norm(Z1)==0)) < 1e-10

    end

    @testset "apply without derivatives" begin
        theta = initTheta(L)
        Y     = randn(nFeatIn(L),nex)
        Z     = apply(L,theta,Y,false)
    end


    @testset "derivative w.r.t Y test" begin

        theta = initTheta(L)
       Y     = randn(nFeatIn(L),nex)

       function testFun(x,v=[])
           if !(isempty(v))
               Z = apply(L,theta,x,true)
               return Z[2], reshape(JYmv(L,v,theta,x,Z[3])[2],size(Z[2]))
           else
               return apply(L,theta,x)[2]
           end
       end
       chkDer, = checkDerivative(testFun,copy(Y),out=out)
       @test chkDer
    end

    @testset "adjoint w.r.t. Y test" begin
    theta = initTheta(L)
       Y     = randn(nFeatIn(L),nex)
       dY    = randn(nFeatIn(L),nex)
       Z     = randn(nFeatOut(L),nex)

       tmp = apply(L,theta,Y,true)
       Z1 =  JYmv(L,copy(dY),theta,copy(Y),tmp[3])[2]
       Z2 =  JYTmv(L,copy(Z),[],theta,copy(Y),tmp[3])

       t1 = dot(Z1,Z)
       t2 = dot(Z2,dY)
       @test abs(t1-t2)/abs(t2) < 1e-8

    end

    if nTheta(L)>0
        @testset "derivative w.r.t theta test" begin
        theta = initTheta(L)
          Y     = randn(nFeatIn(L),nex)

          function testFunTh(x,v=[])
              if !(isempty(v))
                  Z = apply(L,x,copy(Y),true)
                  return Z[2], reshape(Jthetamv(L,v,x,copy(Y),Z[3])[2],size(Z[2]))
              else
                  return apply(L,x,copy(Y))[2]
              end
          end
          chkDer, = checkDerivative(testFunTh,copy(theta),out=out)
          @test chkDer
        end

        @testset "adjoint w.r.t. theta test" begin
        theta = initTheta(L)
          Y     = randn(nFeatIn(L),nex)
          dtheta    = randn(nTheta(L))
          Z     = randn(nFeatOut(L),nex)

          tmp = apply(L,theta,copy(Y),true)
          Z1 =  Jthetamv(L,copy(dtheta),copy(theta),copy(Y),copy(tmp[3]))[2]
          Z2 =  JthetaTmv(L,copy(Z),[],theta,copy(Y),tmp[3])

          t1 = dot(Z1,Z)
          t2 = dot(Z2,dtheta)
          if out
              println("t1=$t1 \t t2=$t2 \t err = $(t1-t2)")
          end
          @test abs(t1-t2)/abs(t2) < 1e-8
        end
    end
end
