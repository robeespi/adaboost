
% ----------------------------------------------
% build a stump from each component and return the best
% Some variables were changed in order to facilitate debugging

function [stumpcellform] = build_stump(X,y,weights)

dim = size(X,2);
weights = weights/sum(weights); % normalized the weights (if not already)

stumpcellform = cell(dim,1); 
errorstump = zeros(dim,1);
for i=1:dim
  stumpcellform{i} = build_onedim_stump(X(:,i),y,weights);
  stumpcellform{i}.ind = i; 
  errorstump(i) = stumpcellform{i}.werr;
end

[min_errorstump,ind] = min(errorstump);

stumpcellform = stumpcellform{ind(1)}; % return the best stump 

% ----------------------------------------------
% build a stump from a single input component

function [stumpcellform] = build_onedim_stump(x,y,weights) 

 % ascending index-xsorter order
[xsorted,I] = sort(x);

% descending index 
Ir = I(end:-1:1); 

%acumulativesum weight of each sample times y train in descending order
score_left  = cumsum(weights(I).*y(I)); % left to right sums
score_right = cumsum(weights(Ir).*y(Ir));  % right to left sums

% score the -1 -> 1 boundary between successive points 
score = -score_left(1:end-1) + score_right(end-1:-1:1); 

% find distinguishable points (possible boundary locations) using linear
% indexing

Idec = find( xsorted(1:end-1)<xsorted(2:end) );

% locate the boundary or give up

if (isempty(Idec)~=0)
  [maxscore,ind] = max(abs(score(Idec))); % maximum weighted agreement
  ind = Idec(ind(1)); 

  stumpcellform.werr = 0.5-0.5*maxscore; % weighted error
  stumpcellform.x0   = (xsorted(ind)+xsorted(ind+1))/2; % threshold
  stumpcellform.s    = sign(score(ind)); % direction of -1 -> 1 change
else
  stumpcellform.werr = 0.5;
  stumpcellform.x0   = 0; 
  stumpcellform.s    = 1; 
end
