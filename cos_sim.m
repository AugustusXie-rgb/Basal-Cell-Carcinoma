% cos_sim

function Sim=Cos(A,B)
    A=A(:);
    B=B(:);
    no=sum(A.*B);
    de=sum(A.*A)*sum(B.*B);
    Sim=no/de;
end
