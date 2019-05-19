Implement stochastic gradient descent in your back propagation program. We will do two types of SGD search. The
original one has the following pseudocode:

I. Original SGD algorithm:

Initialize random weights
While(!converged):
	Shuffle the rows (or row indices)
	for j = 0 to rows-1:
		Determine gradient using just datapoint xj
		Update weights with gradient
	Recalculate objective

II. Mini-batch SGD algorithm:

Initialize random weights
While(!converged):
	Shuffle the rows (or row indices)
	for j = 0 to rows-1:
		Select the first k datapoints where k is the mini-batch size
		Determine gradient using just the selected k datapoints
		Update weights with gradient
	Recalculate objective

Your input, output, and command line parameters are the same as assignment 3
and we do just three nodes in the hidden layer. We leave the offset for
the final layer to be zero at this time.

Test your program on the XOR dataset:

1 0 0
1 1 1
-1 0 1
-1 1 0

1. Test your program on breast cancer and ionosphere given on the website. Is the 
mini-batch faster or the original one? How about accuracy?

2. Is the search faster or more accurate if you keep track of the best objective
in the inner loop?
