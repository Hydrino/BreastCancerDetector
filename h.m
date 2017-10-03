function hi = h(X,theta)
	z = theta'*X';
	z = z';
	
	hi = sigmoid(z);
	
endfunction