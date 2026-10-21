"""Real disposable Chromium + fake decision provider. NOT a model-quality benchmark."""
import asyncio,copy,functools,http.server,json,os,shutil,socket,subprocess,threading,time,urllib.request
from pathlib import Path
import pytest
from playwright.async_api import async_playwright
from decision_deploy.cdp import page_target_id,run_on_page
from decision_deploy.common import read
from decision_deploy.journal import Journal

ROOT=Path(__file__).resolve().parent  # package-scoped imported fixtures

@pytest.fixture(scope='module')
def browser_fixture(tmp_path_factory):
    chrome=os.environ.get('TEST_CHROMIUM') or shutil.which('chromium')
    if not chrome:pytest.skip('no Chromium executable; set TEST_CHROMIUM')
    folder=tmp_path_factory.mktemp('cdp-browser')
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT/'examples')))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    cmd=[chrome,'--headless=new','--disable-dev-shm-usage','--remote-debugging-address=127.0.0.1',f'--remote-debugging-port={port}',f'--user-data-dir={folder}','--no-first-run','about:blank']
    # Root-only container smoke exception; production launcher never disables the sandbox.
    if hasattr(os,'geteuid') and os.geteuid()==0:cmd.append('--no-sandbox')
    process=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    endpoint=f'http://127.0.0.1:{port}'
    ready=False
    for _ in range(100):
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(endpoint+'/json/version',timeout=.2) as f:json.load(f)
            ready=True;break
        except Exception:
            if process.poll() is not None:break
            time.sleep(.05)
    if not ready:
        process.terminate();server.shutdown();raise RuntimeError('Chromium CDP failed to start')
    try:yield endpoint,f'http://127.0.0.1:{server.server_port}'
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        server.shutdown();server.server_close();thread.join(timeout=2)

@pytest.mark.parametrize('case',['complete_and_retry','stale','low_confidence','dry_run','runtime_mismatch'])
def test_live_cdp(browser_fixture,tmp_path,case,monkeypatch):
    endpoint,site=browser_fixture
    in_memory=os.environ.get('TEST_CDP_IN_MEMORY')=='1'
    if in_memory:
        # Synthetic fixture exception ONLY in this test: no production origin policy changes.
        from decision_deploy.common import origin as real_origin
        monkeypatch.setattr('decision_deploy.cdp.origin',lambda url: site if url=='about:blank' else real_origin(url))
    async def scenario():
        async with async_playwright() as p:
            browser=await p.chromium.connect_over_cdp(endpoint)
            page=await browser.contexts[0].new_page()
            if in_memory:await page.set_content((ROOT/'examples/demo.html').read_text(encoding='utf-8'))
            else:await page.goto(site+'/demo.html')
            tid=await page_target_id(page)
            task=copy.deepcopy(read(ROOT/'examples/tasks.json')['tasks']['read_notifications_demo'])
            task.update(allowed_origins=[site],execution_approved=True,expected_runtime_fingerprint='fake-runtime',min_probability=.9)
            calls=[]
            async def predict(state,criteria,obs_id,timeout):
                calls.append(obs_id);key=next(k for k in criteria if k!='needs_host')
                if case=='stale':await page.locator('#app').evaluate("e=>{const p=document.createElement('p');p.textContent='changed';e.appendChild(p)}")
                prob=.55 if case=='low_confidence' else .999
                probs={k:0.0 for k in criteria};probs[key]=prob;probs['needs_host']=1-prob
                return {'runtime_fingerprint':'wrong' if case=='runtime_mismatch' else 'fake-runtime',
                    'answers':{'next_action':{'selected_key':key,'probabilities':probs,'p_selected':prob,'calibrated':True}}}
            j=Journal(tmp_path)
            r=await run_on_page(page,task,predict,j,'fixture-run',tid,dry_run=case=='dry_run')
            if case=='complete_and_retry':
                assert r['status']=='needs_verification' and len(r['events'])==2
                n=len(calls)
                repeated=await run_on_page(page,task,predict,j,'fixture-run',tid,dry_run=False)
                assert repeated==r and len(calls)==n
            elif case=='dry_run':assert r['status']=='dry_run' and len(r['events'])==0
            else:
                assert r['status']=='needs_host' and len(r['events'])==0
                assert await page.locator('#open-settings').is_visible()
            j.close();await page.close()
    asyncio.run(scenario())
