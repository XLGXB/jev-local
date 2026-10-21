import asyncio,json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from decision_deploy.common import probabilities,loopback_url,origin,atoms_for
from decision_deploy.encoding import padding
from decision_deploy.cdp import selected,Stop
from decision_deploy.journal import Journal
from decision_deploy.wrappers import KevGraph,LayaGraph

@pytest.mark.parametrize('url',['http://evil.example:9222','http://127.0.0.1.evil:9222','https://127.0.0.1','http://user:pass@localhost:9222','file:///etc/passwd'])
def test_no_external_endpoint(url):
    with pytest.raises(ValueError):loopback_url(url)
@pytest.mark.parametrize('url',['http://127.0.0.1:9222','http://localhost:8787','ws://127.0.0.1:9222/devtools/browser/x'])
def test_loopback(url):assert loopback_url(url)==url

def test_origin_port():
    assert origin('https://example.test:443/a')=='https://example.test'
    assert origin('https://example.test:8443/a')=='https://example.test:8443'

def test_probabilities():
    p=probabilities([1000,1001],2)
    assert np.isclose(p.sum(),1) and p[1]>p[0]
@pytest.mark.parametrize('z,t',[([1,float('nan')],1),([1,2],0),([1],1),([1,2],float('inf'))])
def test_invalid_probability(z,t):
    with pytest.raises(ValueError):probabilities(z,t)

def test_no_input_labels():
    with pytest.raises(ValueError):atoms_for({'state':{},'questions':{'q':{'type':'noul','instructions':'x','label':True}}})

def test_padding_and_reject():
    v=padding('laya',([1,3,4,2],[1,2]),8,4,0)
    assert v['marker_mask'].tolist()==[[True,True,False,False]]
    assert v['attention_mask'].sum()==4
    with pytest.raises(ValueError):padding('laya',([1,2,3],[1,2]),2,2,0)

def reply(probs=None,key='a',calibrated=True):
    probs=probs or {'a':.99,'needs_host':.01}
    return {'runtime_fingerprint':'r','answers':{'next_action':{'selected_key':key,'probabilities':probs,'p_selected':probs.get(key,0),'calibrated':calibrated}}}

def test_selection():
    assert selected(reply(),{'a':'x','needs_host':'y'},{'expected_runtime_fingerprint':'r','min_probability':.95},False)=='a'
@pytest.mark.parametrize('change',['runtime','candidate','uncalibrated','low','unknown','nan'])
def test_fail_closed(change):
    r=reply();t={'expected_runtime_fingerprint':'r','min_probability':.95}
    if change=='runtime':r['runtime_fingerprint']='bad'
    if change=='candidate':r['answers']['next_action']['probabilities']={'fake':1.0}
    if change=='uncalibrated':r['answers']['next_action']['calibrated']=False
    if change=='low':t['min_probability']=.999
    if change=='unknown':r['answers']['next_action']['selected_key']='invented'
    if change=='nan':r['answers']['next_action']['probabilities']['a']=float('nan')
    with pytest.raises(Stop):selected(r,{'a':'x','needs_host':'y'},t,False)

def test_journal_restart_unknown(tmp_path):
    j=Journal(tmp_path);assert j.begin('r1','fingerprint') is None
    j.reserve('r1','obs','click');j.close();j=Journal(tmp_path)
    assert j.begin('r1','fingerprint')['status']=='outcome_unknown';j.close()

def test_journal_reuse(tmp_path):
    j=Journal(tmp_path);j.begin('r','f');ans={'status':'needs_verification'};j.finish('r',ans)
    assert j.begin('r','f')==ans
    with pytest.raises(ValueError):j.begin('r','different')
    j.close()

def test_kev_pointer_preserved():
    torch.manual_seed(1)
    class Backbone(nn.Module):
        def __init__(self):super().__init__();self.emb=nn.Embedding(20,8)
        def forward(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.emb(input_ids))
    class Head(nn.Module):
        def __init__(self):super().__init__();self.q=nn.Linear(8,4);self.k=nn.Linear(8,4);self.scale=.5;self.temperature=2
    model=SimpleNamespace(lm=Backbone(),head=Head(),option_isolation=False)
    wrapper=KevGraph(model)
    ids=torch.tensor([[1,2,3,4,5]]);opts=torch.tensor([[1,3,0]]);mask=torch.tensor([[True,True,False]])
    out=wrapper(ids,torch.ones_like(ids),torch.arange(5)[None],torch.tensor([[4]]),opts,mask)
    h=model.lm.emb(ids)[0];ref=(model.head.k(h[[1,3]])@model.head.q(h[4]))*.5
    torch.testing.assert_close(out[0,:2],ref)
    assert out[0,2]==-1e4 and model.head.temperature==1

def test_kev_reject_isolation():
    with pytest.raises(ValueError):KevGraph(SimpleNamespace(lm=nn.Identity(),head=nn.Identity(),option_isolation=True))

def test_laya_wrapper_uses_native_logits():
    class Native(nn.Module):
        def forward(self,ids,att,pos,mask,kind):return ids.gather(1,pos).float().masked_fill(~mask,-1e4),torch.ones((1,2))
    g=LayaGraph(Native());out=g(torch.tensor([[5,7,9]]),torch.ones(1,3),torch.tensor([[1,2]]),torch.ones(1,2,dtype=torch.bool),torch.tensor([0]))
    assert out.tolist()==[[7,9]]

@pytest.mark.parametrize("mutation",["duplicate","reserved","origin_path","bad_threshold","approval_type"])
def test_profile_validation(mutation):
    from decision_deploy.cdp import validate_task
    from decision_deploy.common import read
    from pathlib import Path
    task=read(Path(__file__).resolve().parent/"examples/tasks.json")["tasks"]["read_notifications_demo"]
    if mutation=="duplicate":task["actions"].append(task["actions"][0])
    if mutation=="reserved":task["actions"][0]["id"]="needs_host"
    if mutation=="origin_path":task["allowed_origins"]=["https://example.test/path"]
    if mutation=="bad_threshold":task["min_probability"]=float("nan")
    if mutation=="approval_type":task["execution_approved"]="false"
    with pytest.raises(ValueError):validate_task(task)
