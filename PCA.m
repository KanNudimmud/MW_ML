%% PCA
%% Create Data
N = 1000;

% Data
x = [ 1*randn(N,1) .4*randn(N,1) ];

% Rotation matrix
th = pi/4;
R1 = [ cos(th) -sin(th); sin(th) cos(th) ];

% Rotate data
y = x*R1;

% Plot
figure(1)
subplot(121)
axlim = [-1.1 1.1]*max(abs(y(:))); % axis limits
plot(y(:,1),y(:,2),'k.','markersize',10)
set(gca,'xlim',axlim,'ylim',axlim,'xtick',[],'ytick',[])
xlabel('X_1'), ylabel('X_2')
axis square
title('Data space')

%% PCA
% PCA using built-in function
[coeffs,pcscores] = pca(y);

% Plot
subplot(122)
plot(pcscores(:,1),pcscores(:,2),'k.','markersize',10)
set(gca,'xlim',axlim,'ylim',axlim,'xtick',[],'ytick',[])
xlabel('PC_1'), ylabel('PC_2')
axis square
title('PC space')

%% Dimension Reduction

spikes = csvread('spikes.csv');

figure(2)
subplot(511)
plot(mean(spikes,1))

subplot(5,1,2:5)
imagesc(spikes)
xlabel('Time points')

% PCA
[coeffs,pcscores,~,~,explVar] = pca(spikes);

% Show the scree plot (a.k.a. eigenspectrum)
figure(3)
subplot(221)
plot(explVar,'kp-','markerfacecolor','k','markersize',15)
xlabel('Component number')
ylabel('Percent variance explained')

subplot(222)
plot(cumsum(explVar),'kp-','markerfacecolor','k','markersize',15)
xlabel('Component number')
ylabel('Cumulative percent variance explained')


% Show the PC weights for the top two components
subplot(212)
plot(coeffs(:,1:2),'linew',3)
xlabel('Time')
legend({'Comp 1';'Comp 2'})
title('PC weights (coefficients)')


% Show the PC scores
figure(4)
plot(pcscores(:,1),pcscores(:,2),'k.','markersize',2)
xlabel('PC_1'), ylabel('PC_2')
axis square
title('PC space')

%% end.