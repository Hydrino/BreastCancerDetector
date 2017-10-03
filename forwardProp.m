function [h a2] = forwardProp(X,theta1,theta2)

	a1=X;
	z2 = theta1*X';
	a2 = sigmoid(z2);	
	a2 = [ones(1,size(a2,2));a2];

	z3 = theta2*a2;
	h = sigmoid(z3);	

endfunction