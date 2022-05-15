%% Cosine similarity

% Range of requested correlation coefficients
rs = linspace(-1,1,100);

% Sample size
N = 500;

% Initialize output matrix
corrs = zeros(length(rs),2);

for ri=1:length(rs)
    % Generate data
    x = randn(N,1);
    y = x*rs(ri) + randn(N,1)*sqrt(1-rs(ri)^2);
    
    % optional mean-off-centering
    %x = x+10;
    %y = y+10;
    
    % Compute correlation
    corrs(ri,1) = corr(x,y);
    
    % Compute cosine similarity
    cs_num = sum(x.*y);
    cs_den = sqrt(sum(x.*x)) * sqrt(sum(y.*y));
    corrs(ri,2) = cs_num / cs_den;
    
    % using built-in distance function
    %corrs(ri,2) = 1-pdist([x y]','cosine');
end

% Visualize
figure(1)
subplot(121)
plot(rs,corrs,'s-','markerfacecolor','w')
axis square
legend({'Correlation';'Cosine sim.'})
xlabel('Requested correlation')
ylabel('Empirical correlation')

subplot(122)
plot(corrs(:,1),corrs(:,2),'ks','markerfacecolor','w')
axis square
xlabel('Correlation')
ylabel('Cosine similarity')

% Empirical correlation
corr(corrs)

%% end.