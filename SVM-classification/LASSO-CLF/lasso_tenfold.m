function [acc_m,sen_m,spc_m,AUC,auc_curve,acc_max_index] = lasso_tenfold(data,label)
acc = zeros(100,1);
sen = zeros(100,1);
spc = zeros(100,1);
acc_max_index=zeros(100,1);
for i = 1:10
    indices = crossvalind('KFold',label,10);
    for j = 1:10
        test = data(indices == j,:);
        train = data(indices ~= j,:);
        test_label = label(indices == j,1);
        train_label = label(indices ~= j,1);
        [acc1,sen1,spc1,dec_value,index_final]=getBestAcc_l(train,test,train_label,test_label);
        
        acc((i-1)*10+j)=acc1;
        sen((i-1)*10+j)=sen1;
        spc((i-1)*10+j)=spc1;
        dv(indices==j,i)=dec_value;
        acc_max_index((i-1)*10+j,1)=index_final;
    end
end
acc_m=mean(acc);
sen_m=mean(sen);
spc_m=mean(spc);
dv_all_m=mean(dv,2);
[AUC,auc_curve]=ROC(dv_all_m(:,1),label,1,0);

end