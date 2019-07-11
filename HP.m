function [Output]=HP(Input,Mode)
if strcmp(Mode,'learning')==1
    Data_Train=Input.Data_Train;
    Label_Train=Input.Label_Train;
    LayerNum=Input.LayerNum;
    Output.Syst=[];
    uniquelabel=unique(Label_Train);
    CC=length(uniquelabel);
    Output.Syst.Classes=uniquelabel;
    Output.Syst.LayerNum=LayerNum;
    [~,W]=size(Data_Train);
    Xnorm = sqrt(sum(Data_Train.^2, 2));
    Data  = Data_Train ./ Xnorm(:,ones(1,W));
    or=pi/2;
    for cc=1:1:CC
        seq1=Label_Train==uniquelabel(cc);
        data=Data(seq1,:);L1=length(data(:,1));
        Parm=[];
        for ii=1:1:LayerNum
            R0(ii)=1-cos(or*0.5^(ii-1));
            Parm(ii).IDXC(1).Centre=data(1,:);
            Parm(ii).IDXC(1).Support=1;
            Parm(ii).IDXC(1).NoC=1;
            Parm(ii).IDXC(1).IDX=1;
            Parm(ii).IDXC(1).Radius=R0(ii);
            Parm(ii).IDXC(1).Gmean=data(1,:);
            Parm(ii).IDXC(1).NumD=1;
            Parm(ii).DIDX(1)=1;
            Parm(ii).NumC=1;
        end
        for ii=2:1:L1
            indexG1=0;
            indexG=1;
            for jj=1:1:LayerNum
                if indexG1==0
                    Parm(jj).IDXC(indexG).NumD=Parm(jj).IDXC(indexG).NumD+1;
                    [value,position]=min((pdist2(data(ii,:),Parm(jj).IDXC(indexG).Centre,'euclidean')));
                    value=value.^2;
                    if value>2*Parm(jj).IDXC(indexG).Radius(position) %|| (dist1>max(dist2)) || (dist1<min(dist2))
                        Parm(jj).IDXC(indexG).Centre=[Parm(jj).IDXC(indexG).Centre;data(ii,:)];
                        Parm(jj).IDXC(indexG).NoC=Parm(jj).IDXC(indexG).NoC+1;
                        Parm(jj).IDXC(indexG).Support=[Parm(jj).IDXC(indexG).Support;1];
                        Parm(jj).IDXC(indexG).Radius=[Parm(jj).IDXC(indexG).Radius;R0(jj)];
                        Parm(jj).NumC=Parm(jj).NumC+1;
                        Parm(jj).IDXC(indexG).IDX=[Parm(jj).IDXC(indexG).IDX;Parm(jj).NumC];
                        Parm(jj).DIDX=[Parm(jj).DIDX;Parm(jj).NumC];
                        indexG1=1;
                    else
                        Parm(jj).IDXC(indexG).Centre(position,:)=Parm(jj).IDXC(indexG).Centre(position,:)*(Parm(jj).IDXC(indexG).Support(position)/(Parm(jj).IDXC(indexG).Support(position)+1))+data(ii,:)/(Parm(jj).IDXC(indexG).Support(position)+1);
                        Parm(jj).IDXC(indexG).Centre(position,:)=Parm(jj).IDXC(indexG).Centre(position,:)./sqrt(sum(Parm(jj).IDXC(indexG).Centre(position,:).^2,2));
                        Parm(jj).IDXC(indexG).Support(position)=Parm(jj).IDXC(indexG).Support(position)+1;
                        Parm(jj).DIDX=[Parm(jj).DIDX;Parm(jj).IDXC(indexG).IDX(position)];
                        indexG1=0;
                    end
                    indexG=Parm(jj).DIDX(end);
                else
                    Parm(jj).NumC=Parm(jj).NumC+1;
                    Parm(jj).DIDX=[Parm(jj).DIDX;Parm(jj).NumC];
                    Parm(jj).IDXC(indexG).Centre=data(ii,:);
                    Parm(jj).IDXC(indexG).Support=1;
                    Parm(jj).IDXC(indexG).NoC=1;
                    Parm(jj).IDXC(indexG).IDX=Parm(jj).NumC;
                    Parm(jj).IDXC(indexG).Radius=R0(jj);
                    Parm(jj).IDXC(indexG).Gmean=data(ii,:);
                    Parm(jj).IDXC(indexG).NumD=1;
                    indexG=Parm(jj).NumC;
                    indexG1=1;
                end
            end
        end
        Param1{cc}=Parm;
    end
    Output.Syst.Param=Param1;
end
if strcmp(Mode,'testingB')==1
    Data_Test=Input.Data_Test;
    Syst=Input.Syst;
    LayerNum=Syst.LayerNum;
    CC=length(Syst.Classes);
    data=Data_Test;
    [L,W]=size(data);
    data=data./repmat(sqrt(sum(data.^2,2)),1,W);
    score=zeros(L,CC);
    for jj=1:1:L
        for ii=1:1:CC
            indx=1;
            for kk=1:1:LayerNum
                [a,b]=min(pdist2(data(jj,:),Syst.Param{ii}(kk).IDXC(indx).Centre));
                indx=Syst.Param{ii}(kk).IDXC(indx).IDX(b);
            end
            score(jj,ii)=a;
        end
    end
    [~,q]=max(exp(-1*(score).^2),[],2);
    Output.ConfidenceScores=score;
    Output.Labels=Syst.Classes(q);
end
if strcmp(Mode,'testingA')==1
    Data_Test=Input.Data_Test;
    Syst=Input.Syst;
    LayerNum=Syst.LayerNum;
    CC=length(Syst.Classes);
    data=Data_Test;
    [L,W]=size(data);
    data=data./repmat(sqrt(sum(data.^2,2)),1,W);
    CT={};
    for ii=1:1:CC
        CT{ii}=[];
        for kk=1:1:length(Syst.Param{ii}(LayerNum).IDXC)
            CT{ii}=[CT{ii};Syst.Param{ii}(LayerNum).IDXC(kk).Centre];
        end
    end
    score=zeros(L,CC);
    for jj=1:1:L
        for ii=1:1:CC
            [score(jj,ii),~]=min(pdist2(data(jj,:),CT{ii}));
        end
    end
    [~,q]=max(exp(-1*(score).^2),[],2);
    Output.ConfidenceScores=score;
    Output.Labels=Syst.Classes(q);
end
end