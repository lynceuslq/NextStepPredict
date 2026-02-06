import argparse
import os
import os.path
import sys
import numpy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import DistanceMetric
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import random
import math
from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)
from Bio import SeqIO
import torch
import numpy as np

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_embeddings=True, return_hidden_states=True
)

#layerindex=[35]

def embed_prot(sequence,model,layerindex,device,EMBEDDING_CONFIG):
    print("Performing embedding with model: "+model+" on device: "+device+", distance analysis on layer: "+str(layerindex))
    protein = ESMProtein(sequence=sequence)
    client = ESMC.from_pretrained(model).to(device) # or "cpu"
    protein_tensor = client.encode(protein)
    logits_output = client.logits(protein_tensor, EMBEDDING_CONFIG)
    hy=logits_output.hidden_states[layerindex,0,:,:].detach().to(torch.float).cpu()
    return hy


def mutant_loc(seq1, seq2):
    a=list(seq1)
    b=list(seq2)
    diflist=[]
    if len(a) == len(b):
        for i in range(len(a)):
            if a[i] != b[i]:
                diflist.append(i+1)

        return diflist, [seq1[x-1] for x in diflist],  [seq2[x-1] for x in diflist]



def warp_start_loc_todict(start_loc,mmseq):
    start_locdict={}
    if ';' in list(start_loc):
        start_loclist=start_loc.split(';')
        if len(start_loclist) >0:
            for item in start_loclist:
                if ':' in list(item):
                    item=item.split(':')
                    if item[1] in ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']:
                        if int(item[0]) <= len(mmseq):
                            start_locdict.update({int(item[0]):[item[1]]})

    return(start_locdict)

def omit_seq_bystep(sub_dict_bystep,sequence,dict_aln):
    oriseq=list(sequence)
    acc=[]
    for i in sub_dict_bystep.keys():
        for j in sub_dict_bystep[i]:
            loc=dict_aln[i]['aa_post']
            oriseq[loc-1] = j
            acc.append(str(i)+j)

    return {'abbr':"_".join(acc),
            'sub_log':sub_dict_bystep,
            'seq':''.join(oriseq)}
    

def generate_dict_aln(file:str,init_seq:str,freq:int):
    alignments = [''.join(seq.split("\n")[1:]) for seq in open(file).read().split(">")[1:]]
    acclist=[(seq.split("\n")[0]).split(".")[0] for seq in open(file).read().split(">")[1:]]
    accdict={acclist[i]:i for i in range(len(acclist))}
    
    if init_seq not in list(accdict.keys()):
        sys.exit('Cannot find init_seq_accession in the alignment file, please check your inputs!')
    
    nracc=accdict[init_seq]
    aln_post=[]

    for i in range(len(alignments[nracc])):
        if(alignments[nracc][i] != '-'):
            aln_post.append(i+1)

    dict_aln={}

    for j in range(len(aln_post)):
        aa_post=[i+1 for i in range(len(alignments[nracc].replace('-','')))][j]
        dict_aln[aa_post] = {
        "aa_type":[alignments[nracc].replace('-','')[i] for i in range(len(alignments[nracc].replace('-','')))][j],
        "aa_post":aa_post,
        "aln_post":aln_post[j]
        }
    
    lenaln=len(alignments)    
    for i in dict_aln.keys():
        p=dict_aln[i]['aln_post']
        aalist={}
        for j in set([x[p-1] for x in alignments]):
            aalist.update({
            j:count_aa(j,[x[p-1] for x in alignments])/lenaln*100
            })
        dict_aln[i].update({
            'aa_perc':aalist
        })
    
    for i in dict_aln.keys():
        sub_aa={}
        for x in dict_aln[i]['aa_perc']:
            if dict_aln[i]['aa_perc'][x] >= freq and x != dict_aln[i]['aa_type']:
                sub_aa.update({x:dict_aln[i]['aa_perc'][x]})
    
        dict_aln[i].update({
            'aa_sub_cand':sub_aa
        })

    return dict_aln


def count_aa(aa,aln):
    count=0
    for i in range(len(aln)):
        if aln[i] == aa:
            count=count+1

    return(count)

def generate_sub_dict(dict_aln):
    sub_aa_dict={}
    for i in dict_aln.keys():
        if dict_aln[i]['aa_sub_cand'] != {} and list(dict_aln[i]['aa_sub_cand'].keys()) != ['-']:
            sub_aa_dict.update({i:dict_aln[i]})

    sub_dict={}
    for i in sub_aa_dict.keys():
        aalist=list(sub_aa_dict[i]['aa_sub_cand'].keys())
        if '-' in aalist:
            aalist.remove('-')
            
        sub_dict.update({sub_aa_dict[i]['aa_post']:aalist})
    
    return sub_dict


def generate_preddf(embs,formerstate_embs,formerstate_preddf,pred):
    matchacc=[[embs[x]['abbr'],embs[x]['former']] for x in embs.keys()]

    former=[embs[x]['former'] for x in embs.keys()]
    former_val=[float(formerstate_preddf[formerstate_preddf['acc']==x]['omit_val']) for x in former]
    preddf=pd.DataFrame({
    'acc':[embs[x]['abbr'] for x in embs.keys()],
    'former':former,
    'pred':pred,
    'former_pred':former_val,
    })
    preddf['omit_val'] = preddf['pred'] + preddf['former_pred']
    return(preddf)

def estich(val1):
    if val1 < 0.001:
        return(0)
    elif val1 < 0.3:
        return(1) 
    elif val1 < 1:
        return(2)  
    else:
        return(3)

def generate_changedict(embs,formerstate_embs,matchacc):
    changedict={}
    for i in range(len(matchacc)):
        argval=embs[matchacc[i][0]]['hidden_state'][0,:,:].to('cuda')
        formerstate=formerstate_embs[matchacc[i][1]]['hidden_state'][0,:,:].to('cuda')
        ch={}
        mydata = argval-formerstate
        mydata=  mydata.detach().to(torch.float).cpu()

        changedict.update({matchacc[i][0]:{
                    'current':matchacc[i][0],
                    'former':matchacc[i][1],
                    'emb_change':mydata
        }})

    return(changedict)

def generate_change_mat(change,index):
    change_mat_val=torch.stack(([change[x]['emb_change'].amax(-1) for x in change.keys()]))
    print(change_mat_val.shape)
    change_mat_val_new = change_mat_val[:,index]
    change_mat_val=torch.stack(([change[x]['emb_change'].amin(-1) for x in change.keys()]))
    change_mat_val_new = torch.cat((change_mat_val_new,change_mat_val[:,index]),dim=1)
    print('Shape of matrix: '+ str(change_mat_val_new.shape))
    return(change_mat_val_new)


def predict_from_model(model,change_mat):
    return(model.predict(change_mat))


def generatepresubdict(seq1,seq2):
    diffloc,aa,ori=mutant_loc(seq1,seq2)
    md={}
    for i in range(len(diffloc)):
        md.update({diffloc[i]:[aa[i]]})

    return(md)


def generateemb(formerembs,selected_sub_dict,selectedloc,wtseq,dict_aln,model='esmc_600m',layer=[35],device='cuda',EMBEDDING_CONFIG=EMBEDDING_CONFIG):
    embnewsteps={}
    for i in selectedloc:
        for j in list(selected_sub_dict.keys()):
            if j not in list(formerembs[i]['sub_log'].keys()):
                for p in selected_sub_dict[j]:
                    mysub=generatepresubdict(formerembs[i]['seq'],wtseq)
                    mysub.update({j:[p]})
                    print(mysub)
                    seqdict=omit_seq_bystep(mysub,wtseq,dict_aln)
                    seqdict.update({'hidden_state':embed_prot(seqdict['seq'],model,layer,device,EMBEDDING_CONFIG)})
                    seqdict.update({'former':formerembs[i]['abbr']})
                    embnewsteps.update({seqdict['abbr']:seqdict})
    return(embnewsteps)

def generate_sub_dict_nextstep(embs,selected):
    selected_sub_dict={}
    for i in selected:
        mysub=embs[i]['sub_log']
        for j in list(mysub.keys()):
            if j in list(selected_sub_dict.keys()):
                ml=selected_sub_dict[j]
                ml.append(mysub[j][0])
            else:
                ml=[]
                ml.append(mysub[j][0])
        
            selected_sub_dict.update({j:list(set(ml))})

    return(selected_sub_dict)


def wt_wrap_up(seq,emb_model,layer,device,EMBEDDING_CONFIG):
    wt_seq={'abbr':'WT',
        'sub_log':{},
        'seq':seq,
    }
    wt_seq.update({'hidden_state':embed_prot(wt_seq['seq'],emb_model,layer,device,EMBEDDING_CONFIG)})
    wt_seq_emb={wt_seq['abbr']:wt_seq}
    return(wt_seq_emb)
    
def findthediff(x,y):
    diffloc=[]
    if len(y) - len(x) ==1:
        for i in y:
            if i not in x:
                diffloc.append(i)

        if len(diffloc) ==1:
            return diffloc[0]