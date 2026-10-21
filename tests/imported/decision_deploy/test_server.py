import asyncio
from types import SimpleNamespace
import httpx
from decision_deploy.server import create_app

class Fake:
    runtime=SimpleNamespace(fingerprint='fake-runtime')
    def predict(self,request):return {'runtime_fingerprint':'fake-runtime','answers':{'q':{
        'selected_key':'true','probabilities':{'false':.1,'true':.9},'p_selected':.9,'calibrated':False}}}

def test_authenticated_inference():
    async def scenario():
        app=create_app(Fake(),'a'*40)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://127.0.0.1') as c:
            assert (await c.get('/health')).status_code==401
            h={'Authorization':'Bearer '+'a'*40}
            assert (await c.get('/health',headers=h)).json()['ready']
            assert (await c.get('/health',headers={**h,'Origin':'https://evil.example'})).status_code==403
            p={'request_id':'r','observation_id':'o','state':{},'questions':{'q':{'type':'noul','instructions':'visible?'}}}
            r=await c.post('/v1/decide',headers=h,json=p)
            assert r.status_code==200 and r.json()['observation_id']=='o'
            raw='{"request_id":"r","request_id":"x"}'
            assert (await c.post('/v1/decide',headers={**h,'Content-Type':'application/json'},content=raw)).status_code==400
            assert (await c.post('/v1/decide',headers=h,json={**p,'state':'x'*150000})).status_code==413
    asyncio.run(scenario())
