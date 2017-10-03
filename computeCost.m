function [J grad] = computeCost(X,theta,y,lambda)
	m = size(X,1);
	grad = zeros(size(theta));

	a = y'*log(h(X,theta));	
	b = (1-y)'*log(1-h(X,theta));
	
	temp = theta;
	temp(1) = 0;
	reg = (lambda/(2*m))*sum(temp.^2);
	
	J = (-a-b)/m + reg;
	
	grad = (1/m)*(X'*(h(X,theta)-y)) + (lambda/m)			*temp;


endfunction