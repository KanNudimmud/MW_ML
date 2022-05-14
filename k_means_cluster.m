%% K-means clustering
%% Create Data
nPerClust = 50;

% Blur around centroid (std units)
blur = 1;

% XY centroid locations
A = [  1 1 ];
B = [ -3 1 ];
C = [  3 3 ];

% Generate data
a = [ A(1)+randn(nPerClust,1)*blur A(2)+randn(nPerClust,1)*blur ];
b = [ B(1)+randn(nPerClust,1)*blur B(2)+randn(nPerClust,1)*blur ];
c = [ C(1)+randn(nPerClust,1)*blur C(2)+randn(nPerClust,1)*blur ];

% Concatanate into a list
data = [a; b; c];

% Show the data
figure(1)
plot(data(:,1),data(:,2),'s','markerfacecolor','k')
title('How k-means sees the data')

%% K-means Clustering

k = 3; % how many clusters?
[groupidx,cents,sumdist,distances] = kmeans(data,k);

% Draw lines from each data point to the centroids of each cluster
figure(2)
lineColors = 'rkbgmrkbgm';
hold on
for i=1:length(data)
    plot([ data(i,1) cents(groupidx(i),1) ],[ data(i,2) cents(groupidx(i),2) ],lineColors(groupidx(i)))
end

% Draw the raw data in different colors
for i=1:k
    plot(data(groupidx==i,1),data(groupidx==i,2),[ lineColors(i) 'o' ],'markerface','w')
end

% Plot the centroid locations
plot(cents(:,1),cents(:,2),'ko','markerface','g','markersize',10);
xlabel('Axis 1'), ylabel('Axis 2')
title([ 'Result of kmeans clustering (k=' num2str(k) ')' ])

% "ground-truth" centers
plot(A(1),A(2),'kp','linew',2,'markersize',20,'markerfacecolor','y')
plot(B(1),B(2),'kp','linew',2,'markersize',20,'markerfacecolor','y')
plot(C(1),C(2),'kp','linew',2,'markersize',20,'markerfacecolor','y')

%% Determining the Appropriate Number of Clusters (Qualitative)

figure(3)
for k=1:6
    % k-means with this number of clusters
    [groupidx,cents] = kmeans(data,k);
    
    % Draw lines from each data point to the centroids of each cluster
    subplot(2,3,k), hold on
    for i=1:length(data)
        plot([ data(i,1) cents(groupidx(i),1) ],[ data(i,2) cents(groupidx(i),2) ],lineColors(groupidx(i)))
    end
    
    % Draw the raw data in different colors
    for i=1:k
        plot(data(groupidx==i,1),data(groupidx==i,2),[ lineColors(i) 'o' ],'markerface','w')
    end
    
    % Plot the centroid locations
    plot(cents(:,1),cents(:,2),'ko','markerface','g','markersize',10);
    set(gca,'xtick',[],'ytick',[])
    title([  num2str(k) ' clusters' ])
end

%% Number of Clusters (Quantative)

ssds = zeros(6,1);
sils = zeros(6,1);

for k=1:7
    [groupidx,cents,ssd] = kmeans(data,k);
    ssds(k) = mean(ssd);
    
    sils(k) = mean( silhouette(data,groupidx) );
end

figure(4)
subplot(211)
plot(ssds,'k^-','markerfacecolor','k','markersize',15)
set(gca,'xtick',1:k,'xlim',[.5 k+.5])
title('The elbow test')


subplot(212)
plot(sils,'k^-','markerfacecolor','k','markersize',15)
set(gca,'xtick',1:k,'xlim',[.5 k+.5])
title('The silhouette test')
xlabel('Number of clusters')

%% Try again in N-D

nPerClust = 50;

% Blur around centroid (std units)
blur = 1;

% XY centroid locations
A = [  1 2  0 ];
B = [ -2 1 -2 ];
C = [  3 3  2 ];

% Generate data
a = bsxfun(@plus,randn(nPerClust,3)*blur,A);
b = bsxfun(@plus,randn(nPerClust,3)*blur,B);
c = bsxfun(@plus,randn(nPerClust,3)*blur,C);

% Concatanate into a list
data = [a; b; c];

% Show the data
figure(5), clf
plot3(data(:,1),data(:,2),data(:,3),'s','markerfacecolor','k')
axis square, grid on
rotate3d on
title('How k-means sees the data')

%% K-means

k = 3; % how many clusters?
[groupidx,cents,sumdist,distances] = kmeans(data,k);

% Draw lines from each data point to the centroids of each cluster
figure(6), clf
lineColors = 'rkbmrkbm';
hold on
for i=1:length(data)
    plot3([ data(i,1) cents(groupidx(i),1) ],[ data(i,2) cents(groupidx(i),2) ],[ data(i,3) cents(groupidx(i),3) ],lineColors(groupidx(i)))
end

% Draw the raw data in different colors
for i=1:k
    plot3(data(groupidx==i,1),data(groupidx==i,2),data(groupidx==i,3),[ lineColors(i) 'o' ],'markerface','w')
end

% Plot the centroid locations
plot3(cents(:,1),cents(:,2),cents(:,3),'ko','markerface','g','markersize',10);
xlabel('Axis 1'), ylabel('Axis 2'), zlabel('Axis 3')
grid on, axis square
title('Result of kmeans clustering')

% "ground-truth" centers
plot3(A(1),A(2),A(3),'kp','linew',2,'markersize',20,'markerfacecolor','y')
plot3(B(1),B(2),B(3),'kp','linew',2,'markersize',20,'markerfacecolor','y')
plot3(C(1),C(2),C(3),'kp','linew',2,'markersize',20,'markerfacecolor','y')

rotate3d on

%% end.