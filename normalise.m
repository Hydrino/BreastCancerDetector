function X_normalised = normalise(X_train)
	X_normalised = zeros(size(X_train));
	n = size(X_train,2);
	
	for i=1:n
		col = X_train(:,i);
		maxi = max(col);
		mini = min(col);
		X_normalised(:,i) = (col - mean(col))/(maxi-				mini);
	endfor
	
	
endfunction