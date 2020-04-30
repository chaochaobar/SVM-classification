function [ x,ftscore ] = selectFeatureByRFE( train,y,z,iter )
     ft=train;
     label(1:y,:)=1;
     label(y+1:y+z,:)=0;
     [ftRank,ftscore] = ftSel_SVMRFECBR_ori(ft,label);
     [~,index1]=ismember(1:iter,ftRank);
     ftRank(index1)=1;
     ftRank(ftRank~=1)=0;
     x=ftRank;
end
     
     