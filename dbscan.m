%% dbscan
%% Create Data

nPerClust = 50;

% XY centroid locations
A = [  1 0 ];
B = [ -1 0 ];

% Generate data
a = [ A(1)+randn(nPerClust,1) A(2)+randn(nPerClust,1) ];
b = [ B(1)+randn(nPerClust,1) B(2)+randn(nPerClust,1) ];

% Concatanate into a list
data = [a; b];
grouplabels = [ ones(nPerClust,1); 2*ones(nPerClust,1) ];

% Group assignment colors
groupcolors = 'br';

% Show the data
figure(1), hold on
plot(data(grouplabels==1,1),data(grouplabels==1,2),'ks','markerfacecolor',groupcolors(1))
plot(data(grouplabels==2,1),data(grouplabels==2,2),'ks','markerfacecolor',groupcolors(2))

%% Compute Distance Matrix
% Initialize
distmat = zeros(nPerClust*2);

% Loop over elements
for i=1:nPerClust*2
    for j=1:nPerClust*2
        distmat(i,j) = sqrt( (data(i,1)-data(j,1)).^2 + (data(i,2)-data(j,2)).^2 );
    end
end

figure(2)
imagesc(distmat), axis square
set(gca,'clim',[0 4])

%% Classify the new point
% K parameter
k = 3;

% Random new point
newpoint = 2*rand(1,2)-1;

figure(1), hold on
plot(newpoint(1),newpoint(2),'ko','MarkerFaceColor','g','markersize',15)

% Compute distance vector
distvec = zeros(nPerClust*2,1);

for i=1:nPerClust*2
    distvec(i) = sqrt( (data(i,1)-newpoint(1)).^2 + (data(i,2)-newpoint(2)).^2 );
end

%% Show the distances
figure(3)
plot(distvec,'s','markerfacecolor','k')
xlabel('Data element index')
ylabel('Distance to new point')

% Sort the distances
[~,sortidx] = sort(distvec,'ascend');

%% Find the group assignment of the nearest neighbors

disp(grouplabels(sortidx(1:k)))
whichgroup = median(grouplabels(sortidx(1:k)));

% Re-plot
figure(1)
plot(newpoint(1),newpoint(2),'ko','MarkerFaceColor',groupcolors(whichgroup))
plot(data(sortidx(1:k),1),data(sortidx(1:k),2),'ko','markersize',10)

%% Using Built-in functions

% Build the model
mdl = fitcknn(data,grouplabels,'NumNeighbors',k);

% Predict the new data value
whichgroupM = predict(mdl,newpoint)

%% end.