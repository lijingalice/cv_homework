This folder has all 3 stages of the project classifications.

FIRST TWO STAGES:

What I have done
1) Commented out the visualization part of The original codes 
2) Did grid search on Network (original network provided, with one more convolution layer, with batchnormalization) lr, optimization method(SGD,Adam), regulerization. The results are saved in jpg
3) The softmax is removed from the original Net classes, since CrossEntropyLoss contains softmax already.
4) All tests are run with 20 epoches. I only have cpu, so every run takes a while.
5) I didn't include lr adjustment for I am testing the effect of lr on the final result.

What I have found:
1) seems regularization is best at 0 (i.e. no regularization), maybe because the dropout layer is sufficient. You can see this finding by opening all the pdfs with lam=0.1 or lam=1.0
2) For SGD, lr=1.0 is too large. Seems lr=0.01 is doing just fine in both the classes and species classifications. There might be a slight advantage of using normalization layer, as it helps converging, but only very slightly.
3) For Adam, there may be some advantages using lr=0.001, but not obvious. Does this mean Adam is not too sensitive to lr?
4) Both SGD and Adam give similar result at the end.


STAGE 3:
What I have done
1) Done all the coding from scratch. The Net is essentially the original net in Stage 1.
2) Performed two types of minimization, one with loss=loss1+loss2*wt, with wt=1.0 and 1.5 (loss1 is for classes and loss2 is for species). The other one will try num_epoches1 first to minimize loss2, then try to minimize loss1

What I have found:
1) seems weight 1.5 is not stable, which is somewhat surprising.
2) No matter which type of minimization I use, it get to about 50% accurary for species and about 70% for classes, the two stage type might be slightly better.


Extra:
It is actually not that intuitive for doing the two stages minimization as freezing parameter is not that straightforward. The testgrad.py shows that the pitfall is the momentum buffer cannot be easily cleared. 
