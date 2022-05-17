%% Euclidean distance for outlier removal
%% Create some data
N = 40;

% 2-dimensional data
d1 = exp(-abs(randn(1,N)*3));
d2 = exp(-abs(randn(1,N)*5));
datamean = [ mean(d1) mean(d2) ];

% Plot the data
figure(1)
subplot(121), hold on
plot(d1,d2,'ko','markerfacecolor','k','markersize',8);
axis square
set(gca,'xtick',[],'ytick',[])
xlabel('Variable x'), ylabel('Variable y')

% Plot the multivariate mean
plot(datamean(1),datamean(2),'kp','markerfacecolor','g','markersize',15);

%% Compute distance of each point to the mean
% Standard distance measure
ds = zeros(N,1);
for i=1:N
    ds(i) = sqrt( (d1(i)-datamean(1))^2 + (d2(i)-datamean(2))^2 );
end

% Convert to z (don't need the original data)
ds = zscore(ds);

% Plot those distances
subplot(122), hold on
plot(ds,'ko','markerfacecolor',[.7 .5 .3],'markersize',12);
axis square
xlabel('Data index'), ylabel('Z distance')

%% Thresholding
% Threshold in standard deviation units
distanceThresh = 2.5;

% Find the offending points
oidx = find(ds>distanceThresh);

% Cross those out
subplot(122)
plot(oidx,ds(oidx),'x','color','r','markersize',20,'linew',5);

subplot(121)
plot(d1(oidx),d2(oidx),'x','color','r','markersize',20,'linew',5);

%% end.