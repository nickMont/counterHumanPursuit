function [x1,P1] = sigmaPointFilterStep(x0,zk,dynFun,P0,H,R0,a,B,K)
%nonlinear dynamics, linear measurement model
%Q0?

numargin=nargin;
if numargin<7
    a=10^-3;
end
if numargin<8
    B=2;
end
if numargin<9
    K=.1;
end
n=length(x0);

S=chol(P0)';
lambda=a^2*(n+K)-n;

chi_pts=zeros(n,1+2*n);
chi_pts(:,1)=x0;

%Generate chi-values and then propagate.
for i=1:n
   chi_pts(:,1+i) = x0+sqrt(n+lambda)*S(:,i);
end
for i=n+1:2*n
    chi_pts(:,1+i)=x0-sqrt(n+lambda)*S(:,i-n);
end

%propagate through f
chi_bar_pts=zeros(n,1+2*n);
for i=1:2*n+1
    chi_bar_pts(:,i)=feval(dynFun,chi_pts(:,i));
end

%Recalculate xbar,Pbar
W0m=lambda/(lambda+n);
Wim=1/(2*(lambda+n));
W0c=lambda/(lambda+n)+1-a^2+B;
Wic=1/(2*(lambda+n));

xbar_sum=zeros(n,1);
for i=1:2*n+1
    if i==1
        Wm=W0m;
    else
        Wm=Wim;
    end
    xbar_sum=xbar_sum+Wm*chi_bar_pts(:,i);
end
x1bar = xbar_sum;

P_sum=zeros(n,n);
for i=1:2*n+1
    if i==1
        Wc=W0c;
    else
        Wc=Wic;
    end
    P_sum=P_sum+ Wc*(chi_bar_pts(:,i)-x1bar)*(chi_bar_pts(:,i)-x1bar)';
end
P1bar=P_sum;

%linear update step
Sk=H*P1bar*H'+R0;
Kk=P1bar*H'*inv(Sk);
x1=x1bar+Kk*(zk-H*x1bar);
P1=(eye(n)-Kk*H)*P1bar;

end

