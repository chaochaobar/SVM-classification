function [acc_m,sen_m,spc_m,prelabel] = lasso_tenfold(f1,f2,f3,f4,label)
acc = zeros(100,1);
sen = zeros(100,1);
spc = zeros(100,1);
prelabel=zeros(120,10);
for i = 1:10
    indices = crossvalind('KFold',label,10);
    for j = 1:10
        test1 = f1(indices == j,:);
        train1 = f1(indices ~= j,:);
        test2 = f2(indices==j,:);
        train2 = f2(indices ~= j,:);
        test3 = f3(indices==j,:);
        train3 = f3(indices ~= j,:);
        test4 = f4(indices==j,:);
        train4 = f4(indices ~= j,:);
        
        test_label = label(indices == j,1);
        train_label = label(indices ~= j,1);
        
        [test_label1,dev1]=getBestAcc_l(train1,test1,train_label,test_label);
        [test_label2,dev2]=getBestAcc_l(train2,test2,train_label,test_label);
        [test_label3,dev3]=getBestAcc_l(train3,test3,train_label,test_label);
        [test_label4,dev4]=getBestAcc_l(train4,test4,train_label,test_label);
        test_label_predict=[test_label1 test_label2 test_label3 test_label4];
        dev_all=[dev1 dev2 dev3 dev4];
        index=find(dev_all==min(dev_all));
        index_1=index(1,1);
        dev_all(1,index_1)=1000;
        giveup=find(dev_all<1000);
        
        decision=mkdecision(test_label_predict,giveup);
        acc1=getacc(decision,test_label);
        num=size(test_label,1);
        numofA=sum(test_label);
        numofB=num-numofA;
        sen1=sum(decision(1:numofA,:))/numofA;
        spc1=1-sum(decision((numofA+1):end,:))/numofB;
        acc((i-1)*10+j)=acc1;
        sen((i-1)*10+j)=sen1;
        spc((i-1)*10+j)=spc1;
        prelabel(indices==j,i)=decision;
    end
end
acc_m=mean(acc);
sen_m=mean(sen);
spc_m=mean(spc);

end


function [decision]=mkdecision(testlabel,giveup)
[x,~]=size(testlabel);
decision=zeros(x,1);
for i=1:x
    len_1=length(find(testlabel(i,:)==1));
    len_0=length(find(testlabel(i,:)==0));
    if len_1>len_0
        decision(i,1)=1;
    end
    if len_1<len_0
        decision(i,1)=0;
    end
    if len_1==len_0
        len_1=length(find(testlabel(i,giveup)==1));
        len_0=length(find(testlabel(i,giveup)==0));
        if len_1>len_0
            decision(i,1)=1;
        end
        if len_0>len_1
            decision(i,1)=0;
        end
    end           
end   
end

function [acc]=getacc(decision, test_label)
numof1=length(find(test_label==1));
pos_right=length(find(decision(1:numof1,1)==1));
neg_right=length(find(decision(numof1+1:end,1)==0));
acc=(pos_right+neg_right)/length(test_label);
end




