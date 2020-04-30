function [b,tvalue]=selectFeatureByTtest(train,train_label, f)

index_A= train_label==1;
index_B= train_label==0;
train_A=train(index_A,:);
train_B=train(index_B,:);
[~,~,~,stat]=ttest2(train_A,train_B);
tvalue=stat.tstat;
t_abs=abs(tvalue);
t_abs_s=sort(t_abs,'descend');
b=ismember(t_abs,t_abs_s(1,1:f));
end
