from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)

from protdesignmodules import *

class byStepOptimize:
    def __init__(self, model, dict_aln:dict):
        self.model = model
        self.dict_aln = dict_aln
        self.sub_dict=generate_sub_dict(dict_aln)
        self.wtseq=''.join([dict_aln[x]['aa_type'] for x in dict_aln.keys()])
        self.EMBEDDING_CONFIG = LogitsConfig(
        sequence=True, return_embeddings=True, return_hidden_states=True
    )

    def showmodel(self):
        print(self.model)

    def loci_in_model(self,selected_loci):
        self.selected_loci=selected_loci
        self.selected_loci_acc = [str(x)+'max' for x in selected_loci] + [str(x)+'min' for x in selected_loci] 
        print("Using selected loci...")
        return self.selected_loci

    def select_from_last_step(self, num=10):
        laststep=list(self.step_emb.keys())[-1]
        selected=[list(self.preddict[laststep].sort_values(by=['omit_val'],ascending=False)['acc'])[i] for i in range(self.top_num)]
        selectdict={}
        for i in selected:
            self.selectdict.update({self.step_emb[laststep]['abbr']:self.step_emb[laststep]['seq']})
        
        return selectdict

    def _remove_indentical(self):
        seqlist={}
        for i in list(self.topseq.keys()):
            for j in self.topseq[i]:
                ms=self.step_emb[i][j]['seq']
                if ms not in [seqlist[x] for x in list(seqlist.keys())]:
                    seqlist.update({j:self.step_emb[i][j]['seq']})
        
        return seqlist
 
    def select_from_every_step(self, num=10):
        steps=list(self.step_emb.keys())
        steps.pop(0)
        self.topseq={}
        for i in steps:
            preddf=self.preddict[i]
            if len(list(preddf['acc'])) > num:
                selected=[list(preddf.sort_values(by=['omit_val'],ascending=False)['acc'])[i] for i in range(num)]
            else:
                selected=list(preddf['acc'])
            self.topseq.update({i:selected})

        return self._remove_indentical()

    def select_from_top_values(self,num=10):
        emlist=list(self.step_emb.keys())
        emlist.pop(0)
        allvals=[]
        allkeys=[]
        allacc=[]
        for i in emlist:
            allacc=allacc+list(self.preddict[i]['acc'])
            allvals=allvals+list(self.preddict[i]['omit_val'])
            allkeys=allkeys+[i for x in range(len(self.preddict[i]['acc']))]

        self._valuedf=pd.DataFrame({
            'acc':allacc,
            'step':allkeys,
            'omit_val':allvals
        })
        if len(list(self._valuedf['acc'])) > num:
            self.topvaluesdf=self._valuedf.sort_values(by=['omit_val'],ascending=False).head(num)
        else:
            self.topvaluesdf=self._valuedf.sort_values(by=['omit_val'],ascending=False)

        self.topseq={}
        for i in list(set(list(self.topvaluesdf['step']))):
            self.topseq.update({i:list(self.topvaluesdf[self.topvaluesdf['step']==i]['acc'])})

        return self._remove_indentical()

    def startsteps(self,num_steps:int,top_num:int,emb_model:str,layer:list, device:str):
        self.emb_model=emb_model
        self.top_num=top_num
        self.device=device
        self.layer=layer
        self.num_steps=num_steps
        self.step_emb={}
        self.step_emb.update({'step0':wt_wrap_up(self.wtseq,self.emb_model,self.layer,self.device,self.EMBEDDING_CONFIG)})
        selected_sub_dict=self.sub_dict
        selected=['WT']
        self.preddict={}
        self.preddict.update({'step0':pd.DataFrame({
            'acc':['WT'],
            'omit_val':[0]
        })})
        for i in range(self.num_steps):
            if i < len(selected_sub_dict) :
                laststep='step'+str(i)
                newstep='step'+str(i+1)
                self.step_emb.update({newstep:generateemb(self.step_emb[laststep],selected_sub_dict,selected,self.wtseq,self.dict_aln) })
                matchacc=[[self.step_emb[newstep][x]['abbr'],self.step_emb[newstep][x]['former']] for x in self.step_emb[newstep].keys()]
                change_dict_step1=generate_changedict(self.step_emb[newstep],self.step_emb[laststep],matchacc)
                preds=predict_from_model(self.model,generate_change_mat(change_dict_step1,self.selected_loci))
                preddf=generate_preddf(self.step_emb[newstep],self.step_emb[laststep],self.preddict[laststep],preds)
                self.preddict.update({newstep:preddf})
                selected=[list(self.preddict[newstep].sort_values(by=['omit_val'],ascending=False)['acc'])[i] for i in range(self.top_num)]
                selected_sub_dict=generate_sub_dict_nextstep(self.step_emb[newstep],selected)
        



    