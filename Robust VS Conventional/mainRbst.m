%%  Robust Gaussian and Non-gaussain Matched Subspace Detection
% Sam Nazari
% this script tests Robust generalized gaussian detection on:
%   1. "synthetic" data
%   2. "RSVPKeyboard" real case data

%% 1 %% SYNTHETIC DATA
clear all;
close all;

% generate synthetic data
S = [2 0; 0 3; 0 1];  % signal subspace
U = [.2 .1; .3 .4; .1 .4];  % interference subspace
K=size(S,1);   % measurment dimention
M = size(S,2);  % signal subspace dimension
L = size(U,2);  % interference subspace dimension
numOfObserv=1000;     %number of observations

%generate random vectors for phi and theta
theta =  randi(3,M,numOfObserv);
phi = randi(3,L,numOfObserv);

targetNumPerc=0.5;    % number of targets percentage
labels=zeros(1,numOfObserv);
targetIdx = randperm(numOfObserv,floor(targetNumPerc*numOfObserv));
labels(targetIdx)=1;
nonTargetIdx=find(labels==0);
theta(:,nonTargetIdx)=0;
noise=randn(K,numOfObserv)*0.1;    %creat AWGN
observations = S*theta + U*phi+ noise;
signalSubsapce=S;
interferenceSubspace=U;

% Conventional generalized gaussian (nominal)
scoresA=nominalDetector(observations,signalSubsapce);

% Robust generalized gaussian (robust)
scoresB=robustDetector(observations,signalSubsapce);

% plot ROC curves of all detectors
[fprA,tprA,~,aucA,~] = perfcurve(labels,scoresA,1);
[fprB,tprB,~,aucB,~] = perfcurve(labels,scoresB,1);

figure('name','detector performance on synthetic data'),
plot(fprA,tprA,'b'), hold on, plot(fprB,tprB,'-.r');
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC curves - synthetic data');
legend('nominal','Robust','location','SouthEast');

% robust vs nominal comparision curves with increase of interference magnitude
thresh =10; %fixed threshold
imag=0:.2:5; % interference magnitude
TPNominal=zeros(1,length(imag));FPNominal=zeros(1,length(imag));
TPRobust=zeros(1,length(imag));TPRobust=zeros(1,length(imag));
figure('name','robust and nominal detectors performance with changing interference magnitude');
for n0=1:length(imag)
    observations = S*theta + U*phi*imag(n0)+ noise;
    
    lambdaNominal=nominalDetector(observations,signalSubsapce);
    lambdaRobust=robustDetector(observations,signalSubsapce);
    
    [X1,Y1] = perfcurve(labels, lambdaNominal,1);
    [X2,Y2] = perfcurve(labels, lambdaRobust,1);
    
    TPNominal(n0) = sum((lambdaNominal(labels==1))>thresh)/(sum((labels==1)));
    FPNominal(n0) = sum((lambdaNominal(labels==0))>thresh)/(sum((labels==0)));
    
    TPRobust(n0) = sum((lambdaRobust(labels==1)>thresh))/(sum((labels==1)));
    FPRobust(n0) = sum((lambdaRobust(labels==0)>thresh))/(sum((labels==0)));
    
    subplot(2,1,1),plot(X1,Y1,'b');
    xlabel('FPR'); ylabel('TPR');
    title('ROC - robust and nominal detectors performance with changing interference magnitude');
    legend('Nominal');
    hold on
    subplot(2,1,2),
    plot(X2,Y2,'r');
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    legend('Robust');
    hold on
end
figure,
subplot(2,1,1),plot(imag,FPNominal,'b',imag,FPRobust,'r-');
xlabel('Interference Magnitude'); ylabel('False positive rate');
title('Synthetic Data Analysis - fixed threshold, fixed noise level');
legend('Nominal','Robust','location','SouthEast');
subplot(2,1,2),plot(imag,TPNominal,'b',imag,TPRobust,'r-');
xlabel('Interference Magnitude'); ylabel('True positive rate');
legend('Nominal','Robust','location','SouthEast');


%% 1 %% RSVPKeyboard data
clear all;
load normalData;
dataVisualization(trialDataN,trialTargetnessN)

load eyeBlinkContaminatedData;
dataVisualization(trialDataA,trialTargetnessA)
[d1,d2,d3]=size(trialDataN);

trialDataN=trialDataN(1:2:end,:,:); % downsampling data

% apply PCA on each channels of data
trialDataN=PCAonEachChannel(trialDataN);

% concatenate all channel in one column
trialDataNv=reshape(trialDataN,(d1/2)*d2,d3);
% trialDataNv=downsample(trialDataNv,2); % downsampling data

kFold=10;
Indices = crossvalind('Kfold', d3,kFold);
scoresA=zeros(1,d3);
scoresB=zeros(1,d3);

for i=1:kFold
    testIdx=find(Indices==i);
    testFold=trialDataNv(:,testIdx);
    trainFoldLabels=trialTargetnessN(testIdx);
    trainIdx=setdiff([1:d3],testIdx);
    trainFold=trialDataNv(:,trainIdx);
    trainFoldLabels=trialTargetnessN(trainIdx);
    
    %%%%%% train %%%%%%%
    targetTemplate=mean(trainFold(:,trainFoldLabels==1),2);
    
    %    signal subspace estimation with pca
    [H, D, V] = svd(trainFold(:,trainFoldLabels==1),0);
    signalSubsapce=H(:,1:end-20);
    
    % Estimate Interference subspace using hyerspectral unmixing using the ATGP algorithm
    %***NOTE*** below function is borrowed from hyperspectral clustering toolbox
    % reference: http://sourceforge.net/projects/matlabhyperspec/
    interferenceSubspace = hyperAtgp(trainFold(:,trainFoldLabels==0),90);
    
    
    %%%%%% test %%%%%%%
    % a. conventional generalized gaussian (nominal)
    scoresA0=nominalDetector(testFold,signalSubsapce);
    
    % b. Robust generalized gaussian (robust)
    scoresB0=robustDetector(testFold,signalSubsapce);
        
    scoresA(testIdx)=scoresA0;
    scoresB(testIdx)=scoresB0;
     
end

% plot ROC curves of all detectors
[fprA,tprA,~,aucA,~] = perfcurve(trialTargetnessN,scoresA,1);
[fprB,tprB,~,aucB,~] = perfcurve(trialTargetnessN,scoresB,1);
figure('name','detector performance on real EEG data'),
plot(fprA,tprA,'b',fprB,tprB,'-.r');
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC curves - RSVPKeyboard EEG data');
legend('nominal','Robust','AMSD','SNR','general','location','SouthEast');