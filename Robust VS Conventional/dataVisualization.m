%% dataVisualization(trialData,trialTargetness)
% This script plots mean of data for each target and nontarget trials.
%
%%
function dataVisualization(trialData,trialTargetness)
% uncomment if you want to load clean data
% load normalData;

% uncomment if you want to load artifact contaminated data
% load eyeBlinkContaminatedData;


fs=256;
RSVPKeyboardParams.windowDuration=0.5;
triggerPartitioner.windowLengthinSamples = round(RSVPKeyboardParams.windowDuration*fs);

wn=(0:(triggerPartitioner.windowLengthinSamples-1))';
%         fs=256;
timeVector=wn/fs*1000;
targetERPs=squeeze(mean(trialData(:,:,trialTargetness==1),3));
nontargetERPs=squeeze(mean(trialData(:,:,trialTargetness==0),3));



%targetERPAxes=axes(ERPPlotsPanel,'Position',[0,0,1,1]);
ymax=max(max(targetERPs(:)),max(nontargetERPs(:)))*1e6;
ymin=min(min(targetERPs(:)),min(nontargetERPs(:)))*1e6;

figure
subplot(2,1,1);
plot(timeVector,nontargetERPs*1e6);
title('Mean distractor ERP (averaged over trials in the calibration data)');
xlabel('Time (ms)');
ylabel('Magnitude (uV)');
ylim([ymin ymax]);
subplot(2,1,2);
plot(timeVector,targetERPs*1e6);
title('Mean target ERP (averaged over trials in the calibration data)');
xlabel('Time (ms)');
ylabel('Magnitude (uV)');
ylim([ymin ymax]);