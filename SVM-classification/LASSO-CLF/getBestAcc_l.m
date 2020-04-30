function [acc,sen,spc,dev_value,index_final]=getBestAcc_l(train,test,train_label,test_label)
result=zeros(246,3);
dv=zeros(size(test_label,1),246);
for iter=1:246
    [x] = selectFeatureByLasso( train,train_label,iter/246);
    slable=find(x~=0);
    train_t=train(:,slable);
    test_t=test(:,slable);
    model = svmtrain(train_label, train_t, '-s 0 -t 0 -b 1');   %-t 2 RBF
    [predictlabel, acc_t,dec_value]= svmpredict(test_label, test_t, model, '-b 1');
    result(iter,1)=acc_t(1);
    num=size(test_label,1);
    numofA=sum(test_label);
    numofB=num-numofA;
    result(iter,2)=sum(predictlabel(1:numofA,:))/numofA;
    result(iter,3)=1-sum(predictlabel((numofA+1):end,:))/numofB;
    dv(:,iter)=dec_value(:,1);
end
acc_max_index= find(result(:,1)==max(result(:,1)));
sen_all_max=result(acc_max_index,2);
index_sen_max=find(sen_all_max==max(sen_all_max));
index_final=acc_max_index(index_sen_max(1,1));
acc=result(index_final,1);
sen=result(index_final,2);
spc=result(index_final,3);
dev_value=dv(:,index_final);
end
