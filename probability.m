%% Compute probabilities
%% The basic formula
% Counts of the different events
c = [ 1 2 4 3 ];

% Convert to probability (%)
prob = 100*c / sum(c);

%% The example of drawing marbles from a jar
% Colored marble counts
blue   = 40;
yellow = 30;
orange = 20;
totalMarbs = blue + yellow + orange;

% Put them all in a jar
jar = cat(1,1*ones(blue,1),2*ones(yellow,1),3*ones(orange,1));

% Now we draw 500 marbles (with replacement)
numDraws = 500;
drawColors = zeros(numDraws,1);

for drawi=1:numDraws
    % Generate a random integer to draw
    randmarble = ceil(rand*numel(jar));
    
    % Store the color of that marble
    drawColors(drawi) = jar(randmarble);
end

% Now we need to know the proportion of colors drawn
propBlue = sum(drawColors==1) / numDraws;
propYell = sum(drawColors==2) / numDraws;
propOran = sum(drawColors==3) / numDraws;

% Plot those against the theoretical probability
figure(1), hold on
bar([ propBlue propYell propOran ])
plot([0.5 1.5],[1 1]*blue/totalMarbs,'b','linew',3)
plot([1.5 2.5],[1 1]*yellow/totalMarbs,'b','linew',3)
plot([2.5 3.5],[1 1]*orange/totalMarbs,'b','linew',3)

set(gca,'xtick',1:3,'XTickLabel',{'Blue';'Yellow';'Orange'})
xlabel('Marble color'), ylabel('Proportion/probability')
legend({'Proportion';'probability'})

%% end.