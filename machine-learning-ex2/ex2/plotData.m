function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos = find(y==1); neg = find(y == 0); % Return indices of X where y==1 or y==0
% Plot Examples 
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2 , 'MarkerSize', 7); % indices in 1st and 2nd  column where y==1
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7); %indices in 1st and 2nd  column where y==0
xlabel('Exam 1 score');
ylabel('Exam 2 score');







% =========================================================================



hold off;

end
