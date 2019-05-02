clear;clc;
c=299792458;planck=6.62607004e-34;eV=1.60217662e-19; %define constants

file_list = {'Run_423_494220989.h5'}; %LDM file list containing the FEL spectra
folder = '/Users/rebernik/Desktop/phase_locked_pulses/'; %folder with LDM files

ij=1; %open just the first file; this can be implemented as a loop to go through all the files in the list
fle_name = file_list{ij};
file1_use= [folder, fle_name]; 
info = hdf5info(file1_use);


Wavelength = '/photon_diagnostics/Spectrometer/Wavelength'; %define paths in the HDF file for reading the spectrum and spectrometer settings
WavelengthSpan = '/photon_diagnostics/Spectrometer/WavelengthSpan';
Pixel2micron = '/photon_diagnostics/Spectrometer/Pixel2micron';
hor_spectrum= '/photon_diagnostics/Spectrometer/hor_spectrum';
    

[dat_Wavelength] = hdf5read(file1_use,Wavelength); %read and use spectrometer settings to calculate the wavelength
[dat_WavelengthSpan] = hdf5read(file1_use,WavelengthSpan);
[dat_Pixel2micron] = hdf5read(file1_use,Pixel2micron);
lambda_use=dat_Wavelength+double([1:1:1000]-500).*dat_Pixel2micron.*1e-3.*dat_WavelengthSpan;%calculated wavelength


[spectrum, attr1] = hdf5read(file1_use,hor_spectrum); %read the spectra and convert to double
spectrum=double(spectrum);
sizespectrum=size(spectrum);
for ijk=1:sizespectrum(2)
    spectrum(:,ijk)=spectrum(:,ijk)/max(spectrum(:,ijk));%normalize spectra; the ijk variable reffers to a specific single shot
end



E0=1.16867e9*eV; %electron beam nominal energy
sigmaE=150e3*eV; %electron beam energy spread
R56=50e-6; %dispersive strength
ebeamlinchirp=0.19e6*eV/(1e-12);% electron beam cubic chirp
ebeamquadchirp=5.42e6*eV/(1e-12)^2;% electron beam quadratic chirp
%ebeamcubechirp=3e6*eV/(1e-12)^3;% electron beam cubic chirp


lambdaseed=252.87e-9; %seed laser wavelength
k1=2*pi/lambdaseed; %seed laser wave number

n=5; %harmonic number
lambdaFEL=lambdaseed/n; %FEL central wavelength


tau10=130e-15; % first seed transform-limited pulse duration
dlambda1=(c/(c/lambdaseed)^2)*(0.44/tau10); %transform-limited bandwidth
GDD1=0e-27; %first seed linear frequency (quadratic phase) chirp
tau1=sqrt(1+(4*log(2)*GDD1/tau10^2)^2)*tau10; %first seed pulse duration


tau20=tau10;% second seed transform-limited pulse duration
dlambda2=(c/(c/lambdaseed)^2)*(0.44/tau20); %transform-limited bandwidth
GDD2=0e-27; %second seed linear frequency (quadratic phase) chirp
tau2=sqrt(1+(4*log(2)*GDD2/tau20^2)^2)*tau20; %second seed pulse duration

deltat=150e-15; %separation between the seeds
deltaphi=pi*0.87; %relative phase between the seeds
ebeamtiming=-400e-15;% relative timing between the electron beam and the two seeds


tmin=-40*max(tau1,tau2)+deltat/2;% time window for calculations
tmax=40*max(tau1,tau2)+deltat/2;
npoints=4000; % number of points in the time window
step=(tmax-tmin)/npoints; %step
time=(tmin:step:tmax);

z=time*c; %longitudinal coordinate in the electron beam

tminplot=-300e-15; %time range for plotting
tmaxplot=400e-15;


Psi1=(1/(2*GDD1+(tau10^4)/(8*(log(2)^2)*GDD1)))*time.^2; %seed phases accounting for the seed chirps
Psi2=(1/(2*GDD2+(tau20^4)/(8*(log(2)^2)*GDD2)))*(time-deltat).^2;


C1=1; %relative seed amplitudes
C2=1;


seedfield=(C1*exp(-2*log(2)*time.^2/tau1^2).*exp(1i*Psi1)+C2*exp(-2*log(2)*(time-deltat).^2/tau2^2).*exp(1i*Psi2)*exp(1i*deltaphi)); %seed electric field; first seed sentered at time=0 fs
seedenvelope=abs(seedfield).^2; %seed envelope
seedphase=unwrap(angle(seedfield)); %seed phase

A0=3; %amplitude of the energy modulation of the electron beam induced by the seeds
A=A0*sqrt(seedenvelope); %temporal profile of the energy modulation of the electron beam
B=R56*k1*sigmaE/E0 %normalized dispersive strength

%electron beam energy profile defined as the nominal energy E0 plus
%quadratic and cubic terms
ebeamenergyprofile=(E0+ebeamlinchirp*(time-ebeamtiming)+(1/2)*ebeamquadchirp*(time-ebeamtiming).^2);%+ebeamcubechirp*(time-ebeamtiming).^3);
ebeamphase=(B/sigmaE)*ebeamenergyprofile; %electorn beam energy profile induces a phase onto the FEL pulse

%bunching (proportional to the FEL electric field) in the time domain
btime=exp(-(n*B)^2/2).*besselj(n, -n*B*A).*exp(1i*n*seedphase).*exp(1i*n*ebeamphase);
maxbunching=max(abs(btime)) %calculates maximum bunching, this depends on the energy modulation amplitude A and R56, should be several percent to correspond to realistic experimental conditions

FELint=sum(abs(btime).^2); %total FEL intensity in a.u.

FELtime=abs(btime).^2/max(abs(btime).^2); %normalized FEL intensity profile in the time domain
FELphase=unwrap(angle(btime));% FEL phase



%transformation into the spectral domain
seedspectralenvelope=fftshift(fft(fftshift(seedfield)));% seed spectrum
seedspectralenvelope=abs(seedspectralenvelope).^2/max(abs(seedspectralenvelope).^2);% normalize the seed spectrum

FELfreq=abs(fftshift(fft(fftshift(btime)))).^2; %FEL spectral envelope
FELfreq=FELfreq/max(FELfreq); %normalize the FEL spectral enevelope

timerange=max(time)-min(time);% time range for FFT
sizetime=size(time);
freqrange=c/(lambdaseed/n)+(-(1/2)*(1/timerange)*(sizetime(2)-1):(1/timerange):(1/2)*(1/timerange)*(sizetime(2)-1));% calculated frequency range for FFT
lambdarange=c./freqrange;%calculated wavelength range for FFT

fsimages=16;%font size in images
figure(111) %plots the seed envelope in the time domain
%hold on
subplot(3,1,1)
plot(time*1e15,seedenvelope,'LineWidth',2)
set(gca,'LineWidth',2);
set(gca,'FontSize',fsimages)
xlabel('TIME (fs)');
ylabel('SEED INTENSITY (a.u.)');
xlim([tminplot*1e15 tmaxplot*1e15])

subplot(3,1,2)  %plots the seed phase in the time domain
%hold on
plot(time*1e15,seedphase,'LineWidth',2)
set(gca,'LineWidth',2);
set(gca,'FontSize',fsimages)
xlabel('TIME (fs)');
ylabel('SEED PHASE');
xlim([tminplot*1e15 tmaxplot*1e15])

subplot(3,1,3)  %plots the seed envelope in the wavelength domain
%hold on
plot(lambdarange*1e9*n,seedspectralenvelope,'LineWidth',2)
%xlim([(0.997)*lambdaseed/n*1e9 (1.003)*lambdaseed/n*1e9])
xlim([252 253.5])
set(gca,'LineWidth',2);
set(gca,'FontSize',fsimages)
xlabel('WAVELENGTH (nm)');
ylabel('SEED SPECTRUM (a.u.)');


figure(222)  %plots the FEL envelope in the time domain
%hold on
subplot(3,1,1)
plot(time*1e15,FELtime,'LineWidth',2)
set(gca,'LineWidth',3);
set(gca,'FontSize',fsimages)
xlabel('TIME (fs)');
ylabel('FEL intensity (a.u.)');
xlim([tminplot*1e15 tmaxplot*1e15])

subplot(3,1,2)  %plots the FEL phase in the time domain
%hold on
plot(time*1e15,FELphase,'LineWidth',2)
set(gca,'LineWidth',2);
set(gca,'FontSize',fsimages)
xlabel('TIME (fs)');
ylabel('FEL PHASE');
xlim([tminplot*1e15 tmaxplot*1e15])


subplot(3,1,3)  %plots the FEL envelope in the time domain
%hold on
plot(lambdarange*1e9,FELfreq,'LineWidth',2)
%xlim([(0.997)*lambdaseed/n*1e9 (1.003)*lambdaseed/n*1e9])
%xlim([-50/1000+c/(lambdaseed/n)*planck/eV 50/1000+c/(lambdaseed/n)*planck/eV])
xlim([50.3 50.7])
set(gca,'LineWidth',2);
set(gca,'FontSize',fsimages)
xlabel('WAVELENGTH (nm)');
ylabel('FEL spectrum (a.u.)');



specnumber=1; %which experimental single shot spectrum to plot (from HDF file)
plot(lambda_use,spectrum(:,specnumber),lambdarange*1e9,FELfreq,'LineWidth',2) %plot the experimental and calculated FEL spectra onto the same plot
legend('experiment','calculation')
xlim([50.3 50.7]) %wavelength range in plots
set(gca,'LineWidth',2);
set(gca,'FontSize',fsimages)
xlabel('WAVELENGTH (nm)');
ylabel('FEL spectrum (a.u.)');

figure(351) %shows the electron beam energy profile
plot(time*1e12,ebeamenergyprofile/eV/1e9,'LineWidth',2)
xlim([-0.8+ebeamtiming 0.8+ebeamtiming])
set(gca,'LineWidth',2);
set(gca,'FontSize',fsimages)
xlabel('time (ps)');
ylabel('electron beam energy (GeV)');


%%%%
%%%%








% 
% for kl=1:200
% ebeamtiming=-1000e-15+kl*10e-15;
% 
% ebeamenergyprofile=(E0+ebeamquadchirp*(time+ebeamtiming).^2+ebeamcubechirp*(time+ebeamtiming).^3);
% ebeamphase=(B/sigmaE)*ebeamenergyprofile;
% %ebeamenergyprofile=0;
% 
% btime=exp(-(n*B)^2/2).*besselj(n, -n*B*A).*exp(1i*n*seedphase).*exp(1i*n*ebeamphase);
% maxbunching=max(abs(btime));
% 
% FELint=sum(abs(btime).^2);
% 
% FELtime=abs(btime).^2/max(abs(btime).^2);
% 
% FELfreq=abs(fftshift(fft(fftshift(btime)))).^2;
% FELfreq=FELfreq/max(FELfreq);
% 
% timerange=max(time)-min(time);
% sizetime=size(time);
% freqrange=c/(lambdaseed/n)+(-(1/2)*(1/timerange)*(sizetime(2)-1):(1/timerange):(1/2)*(1/timerange)*(sizetime(2)-1));
% lambdarange=c./freqrange;%-lambdaseed/n;
% 
% spectraumdelayscan(kl,:)=FELfreq(:);
% ebeamtimingscan(kl)=ebeamtiming;
% 
% end
% 
% figure(433)
% surfc(ebeamtimingscan*1e12,lambdarange*1e9,spectraumdelayscan')
% view(2)
% shading interp
% ylim([50.4 51.2])
% xlim([-0.75 0.75])


deltaphiscan=0;
doscanphase=0;
doscanduration=0;


npointsphase=100;
npointstime=100;
scandeltaphi=(0:2*pi/npointsphase:4*pi);
scanduration=(10e-15:180e-15/npointstime:400e-15);
sizescandeltaphi=size(scandeltaphi);
sizescanduration=size(scanduration);

% if doscanphase==1
% 
%   
% for ijk=1:sizescandeltaphi(2)
% 
%     deltaphi=scandeltaphi(ijk);
% seedfield=(C1*exp(-2*log(2)*time.^2/tau1^2).*exp(1i*Psi1)+C2*exp(-2*log(2)*(time-deltat).^2/tau2^2).*exp(1i*Psi2)*exp(1i*deltaphi));
% seedenvelope=abs(seedfield).^2;
% seedphase=unwrap(angle(seedfield));
% 
% 
% seedspectralenvelope=fftshift(fft(fftshift(seedfield)));
% seedspectralenvelope=abs(seedspectralenvelope).^2/max(abs(seedspectralenvelope).^2);
% 
% scandeltaphiseedspectrum(ijk,:)=seedspectralenvelope;
% 
% A=A0*sqrt(seedenvelope);
% 
% 
% btime=besselj(n, -n*B*A).*exp(1i*n*seedphase);
% %maxbunchingscan=max(abs(btime))
% FELintensity(ijk)=sum(abs(btime).^2);
% FELtime=abs(btime).^2/max(abs(btime).^2);
% FELphase=unwrap(angle(btime));
% 
% FELfreq=abs(fftshift(fft(fftshift(btime)))).^2;
% FELfreq=FELfreq/max(FELfreq);
% 
% timerange=max(time)-min(time);
% sizetime=size(time);
% freqrange=c/(lambdaseed/n)+(-(1/2)*(1/timerange)*(sizetime(2)-1):(1/timerange):(1/2)*(1/timerange)*(sizetime(2)-1));
% lambdarange=c./freqrange-lambdaseed/n;
% 
% fsimages=16;
% 
% 
% scandeltaphispectrum(ijk,:)=FELfreq;
% 
% end
% 
% figure(333)
% imagesc(scandeltaphi,(freqrange-c/(lambdaseed/n))*planck/eV*1e3,scandeltaphispectrum')
% ylim([-50 50])
% set(gca,'LineWidth',2);
% set(gca,'FontSize',fsimages)
% xlabel('FEL PHASE DIFFERENCE');
% ylabel('ENERGY DIFFERENCE (meV)');
% 
% figure(888)
% imagesc(scandeltaphi,(freqrange-c/(lambdaseed/n))*planck/eV*1e3,scandeltaphiseedspectrum')
% ylim([-50 50])
% set(gca,'LineWidth',2);
% set(gca,'FontSize',fsimages)
% xlabel('PHASE DIFFERENCE');
% ylabel('ENERGY DIFFERENCE (meV)');
% 
% figure(333111)
% plot(scandeltaphi,FELintensity,'LineWidth',2)
% set(gca,'LineWidth',2);
% set(gca,'FontSize',fsimages)
% xlabel('PHASE DIFFERENCE');
% ylabel('FEL INTENSITY (a.u.)');
% 
% end
% 
% 
% 
% 
% 
% 
% if doscanduration==1
%   deltaphi=deltaphiscan;
%   clear FELfreq;
% for ijk=1:sizescanduration(2)
% 
%     tau1=scanduration(ijk);
%     tau2=tau1;
%     
% seedfield=(C1*exp(-2*log(2)*time.^2/tau1^2).*exp(1i*Psi1)+C2*exp(-2*log(2)*(time-deltat).^2/tau2^2).*exp(1i*Psi2)*exp(1i*deltaphi));
% seedenvelope=abs(seedfield).^2;
% seedphase=unwrap(angle(seedfield));
% 
% 
% A=A0*sqrt(seedenvelope);
% 
% 
% btime=besselj(n, -n*B*A).*exp(1i*n*seedphase);
% FELintensity(ijk)=sum(abs(btime).^2);
% FELtime=abs(btime).^2/max(abs(btime).^2);
% FELphase=unwrap(angle(btime));
% 
% FELfreq=abs(fftshift(fft(fftshift(btime)))).^2;
% FELfreq=FELfreq/max(FELfreq);
% 
% timerange=max(time)-min(time);
% sizetime=size(time);
% freqrange=c/(lambdaseed/n)+(-(1/2)*(1/timerange)*(sizetime(2)-1):(1/timerange):(1/2)*(1/timerange)*(sizetime(2)-1));
% lambdarange=c./freqrange-lambdaseed/n;
% 
% fsimages=16;
% 
% 
% scandurationspectrum(ijk,:)=FELfreq;
% 
% end
% 
% figure(444)
% imagesc(scanduration*1e15,(freqrange-c/(lambdaseed/n))*planck/eV*1e3,scandurationspectrum')
% ylim([-50 50])
% set(gca,'LineWidth',2);
% set(gca,'FontSize',fsimages)
% xlabel('SEED DURATION (fs)');
% ylabel('ENERGY DIFFERENCE (meV)');
% 
% 
% figure(444111)
% plot(scanduration*1e15,FELintensity,'LineWidth',2)
% set(gca,'LineWidth',2);
% set(gca,'FontSize',fsimages)
% xlabel('SEED DURATION (fs)');
% ylabel('FEL INTENSITY (a.u.)');
% end

