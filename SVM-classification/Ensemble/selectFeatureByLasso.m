function [ coe,funVal] = selectFeatureByLasso( f,y,Lam )

opts=[];
opts=sll_opts(opts);
opts.rFlag=1;
opts.rsL2=0;

[coe,funVal]=LeastR(f,y,Lam,opts);
