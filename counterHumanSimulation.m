clear;clc;
%learns opponent behavior online and develops counterstrategy
%also solves game optimally to determine what benefit is received
%dynamics are single integrator with direct velocity input

nX=2; nU=nX; nV=nX;
A_tr=eye(nX);
B_tr=eye(nX);
Gammak=eye(nX);
Qdynpur=.01*eye(nX); %noise covariance of dynamics propagation
Qdyneva=.01*eye(nX);
Robspur=.01*eye(nX);
Robseva=.01*eye(nX);
Revastack=[Robseva zeros(nX,nX); Robseva zeros(nX,nX)];

nSim=10;

flagPlotAsGo=1;
flagUseDetm=1; %deterministic if ==1

MCL=1;

for iiMCL=1:MCL
    
    xPur=[0;0];
    xEva=[2.5;0];
    
    %pursuer uses greedy controls in repsonse
    Qfpur=.1*eye(nX);
    Rpur=.1*eye(nX);
    %initialize game state plot
    axisSet=[-5 5 -5 5];
    if flagPlotAsGo==1
        figure(1); clf;
        purPlot=plot(xPur(1),xPur(2),'g*');
        hold on
        evaPlot=plot(xEva(1),xEva(2),'r*');
        axis(axisSet);
    end
    
    %evader's true action
    Ktrue0=[.05 0;0 .15];
    Ktrue=zeros(nU,nX,nSim);
    for i=1:nSim
        Ktrue(:,:,i)=Ktrue0;
    end
    uMaxPur=.25;
    uMaxEva=.25;
    
    %stack [xEva;k1..kn] into one vector
    k1_0=.1; k2_0=.1;
    xe_K_estPur=[xEva;k1_0;k2_0];
    %e=xP-xE;
    
    %initial cov parameters
    P_PEeva=.1*eye(2*nX);
    P_Ppur=.1*eye(nX); %pursuer initial self-covariance
    P_EKpur=[.1*eye(nX) zeros(nX,nX); zeros(nX,nX) 1*eye(nX)];
    
    xPpurStore=zeros(nX,nSim+1);
    xEpurStore=zeros(nX,nSim+1);
    xPtrueStore=zeros(nX,nSim+1);
    xEtrueStore=zeros(nX,nSim+1);
    xPpurStore(:,1)=xPur; xEpurStore(:,1)=xEva;
    xPtrueStore(:,1)=xPur; xEtrueStore(:,1)=xEva;
    KvalStore=zeros(nX,nSim+1);
    KvalStore(:,1)=[k1_0;k2_0];
    
    for i=1:nSim
        
        if i==1
            xPpur=xPur;
            xEpur=xEva;
            xPeva=xPur;
            xEeva=xEva;
            xPEeva=[xPur;xEva];
        else
            xPpur=xPpur;
            xEpur=xe_K_estPur(1:nX);
            xPeva=xPEeva(1:nX);
            xEeva=xPEeva(nX+1:2*nX);
        end
        
        %pursuer estimates evader action based on mean estimates for Kmat
        Kmat=diag(xe_K_estPur(nX+1:2*nX));
        uPur0=zeros(nU,1);
        JJ=@(u) [A_tr*xPur+B_tr*u-(A_tr*xEva-B_tr*Kmat*(xPpur-xEpur))]'*Qfpur*...
            [A_tr*xPur+B_tr*u-(A_tr*xEva-B_tr*Kmat*(xPpur-xEpur))]+u'*Rpur*u;
        uPunsat=fminsearch(JJ,uPur0)
        uP=vectorSaturationF(uPunsat,0,uMaxPur)
        
        %evader true behavior
        uE=-vectorSaturationF(Ktrue(:,:,i)*(xPeva-xEeva),0,uMaxEva);
        
        %dynamics propagation
        if flagUseDetm==1
            nP=zeros(nV,1);
            nE=zeros(nV,1);
        else
            nP=chol(Qdynpur)*randn(nV,1);
            nE=chol(Qdyneva)*randn(nV,1);
        end
        xPur=A_tr*xPur+B_tr*uP+Gammak*nP;
        xEva=A_tr*xEva+B_tr*uE+Gammak*nE;
        
        %generate observation vectors
        if flagUseDetm==1
            wPpur=zeros(nX,1);
            wEpur=zeros(nX,1);
            %wPEeva=zeros(2*nX,1);
        else
            wPpur=chol(Robspur)*randn(nX,1);
            wEpur=chol(Robspur)*randn(nX,1);
            %wPEpur=chol(Revastack)*randn(2*nX,1);
        end
        zPpur=xPur+wPpur;
        zEpur=xEva+wEpur;
        %zPEeva=[xPur;xEva]+wPEeva;
        
        %filters
        xPEeva=[xPur;xEva]; %assume that the evader (human player) does not
            %filter the game state.  Iffy model for human behavior.
        [xPpur,P_Ppur]=linearKFStep(xPpur,zPpur,A_tr,B_tr,Gammak,P_Ppur,Qdynpur,uP,eye(nX),Robspur);
        dynFun=@(xk) [A_tr*xk(1:nX)-B_tr*vectorSaturationF(diag(xk(nX+1:2*nX))*(xPeva-xk(1:nX)),0,uMaxEva);xk(nX+1:2*nX)];
        HH=zeros(nX,2*nX);
        [xe_K_estPur,P_EKpur]=sigmaPointFilterStep(xe_K_estPur,zEpur,dynFun,P_EKpur,HH,Robspur);
        
        %plotting
        if flagPlotAsGo==1
            pause(1);
            delete(purPlot); delete(evaPlot);
            purPlot=plot(xPur(1),xPur(2),'g*');
            hold on
            evaPlot=plot(xEva(1),xEva(2),'r*');
            axis(axisSet);
            legend(strcat('t=',num2str(i)));
        end
        
        %store data
        xPpurStore(:,i+1)=xPpur;
        xEpurStore(:,i+1)=xe_K_estPur(1:nX);
        xPtrueStore(:,i+1)=xPur;
        xEtrueStore(:,i+1)=xEva;
        KvalStore(:,i+1)=xe_K_estPur(nX+1:2*nX);
        
    end
    
    
end

% figure(2)
% plot(1:nSim+1,KvalStore(1,:),'r')
% hold on
% plot(1:nSim+1,KvalStore(2,:),'g')
% legend('k1','k2')






