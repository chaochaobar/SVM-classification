function [test_label,dev]=getBestAcc_l(train,test,train_label,test_label)
result=zeros(246,2);
testlabel_all=zeros(size(test_label,1),246);
dev_all=zeros(size(test_label,1),1);
for iter=1:246
    [x] = selectFeatureByLasso( train,train_label,iter/246);
    slable=find(x~=0);
    train_t=train(:,slable);
    test_t=test(:,slable);
    model = svmtrain(train_label, train_t, '-s 0 -t 0 -b 1');   %linear-kernal
    [predictlabel, acc_t,~]= svmpredict(test_label, test_t, model, '-b 1');
    [~,dev_train,~]=svmpredict(train_label, train_t, model, '-b 1');
    dev_all(iter,1)=1-dev_train(1);  %%%training bias
    result(iter,1)=acc_t(1);
    numofA=sum(test_label);
    result(iter,2)=sum(predictlabel(1:numofA,:))/numofA;
    testlabel_all(:,iter)=predictlabel;
end
   acc_max_index= find(result(:,1)==max(result(:,1)));
   sen_all_max=result(acc_max_index,2);
   index_sen_max=find(sen_all_max==max(sen_all_max));
   index_final=acc_max_index(index_sen_max(1,1));
   test_label=testlabel_all(:,index_final);
   dev=dev_all(index_final,1);
end
