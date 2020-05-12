% t=0:1/50:20;
% x=sin(2*pi*t*3);
% x(1:100)=0;
% x(200:350)=0;
% x(400:501)=0;
% [m, mt, mf] = norm_spectrogram(t, x);
% plot_norm_matrix(m, mt, mf, t, x, '1')

clear all;close all;clc;
GMFolder=('GM_Northridge_ReSize_Opensees/');
numGM = 304;
repeat_id=[78 107 230 259]; %delete the same longitude and latitude
i=1;
t=[0.005:0.005:30];
for iGM = 1:numGM
    if ismember(iGM,repeat_id)
        continue
    end 
    GMfile=strcat(GMFolder,num2str(iGM),'.txt');
    acc=load(GMfile);
    [m, mt, mf] = norm_spectrogram(t, acc);
    title=strcat('No-',num2str(i));
    plot_norm_matrix(m, mt, mf, t, acc, title);
    savefile=strcat('..\GM_Spectrogram\','NR_',num2str(i),'.mat');
    save(savefile,'m');
    i=i+1;
end
    
