"""Browser E2E: fully real FastAPI responses, routed to localhost-restricted Chromium."""
from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse
import re
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cnnviz.server import app


def run(out='artifacts/browser'):
    Path(out).mkdir(parents=True,exist_ok=True)
    client=TestClient(app)
    html=Path('web/index.html').read_text(encoding='utf-8')
    html=re.sub(r'<script src="/(?:app|dynamics|concepts).js" defer></script>','',html)
    html=re.sub(r'<link rel="stylesheet" href="/(?:app|dynamics|concepts).css"\s*/>','',html)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
        for name,width,height in [('desktop',1440,900),('mobile',390,844)]:
            page=browser.new_page(viewport={'width':width,'height':height})
            errors=[];page.on('pageerror',lambda err:errors.append(str(err)))
            def transport(source,url):
                u=urlparse(url)
                response=client.get(u.path+('?' + u.query if u.query else ''))
                return {'status':response.status_code,'body':response.text,'content_type':response.headers.get('content-type','application/json')}
            page.expose_binding('backendFetch',transport)
            page.set_content(html,wait_until='domcontentloaded')
            for css in ['web/app.css','web/dynamics.css','web/concepts.css']:
                page.add_style_tag(content=Path(css).read_text(encoding='utf-8'))
            page.evaluate("window.fetch=async function(url){const r=await window.backendFetch(String(url));return new Response(r.body,{status:r.status,headers:{'content-type':r.content_type}})}")
            for js in ['web/app.js','web/dynamics.js','web/concepts.js']:
                page.add_script_tag(content=Path(js).read_text(encoding='utf-8'))
            page.wait_for_function("document.getElementById('cx-gallery')?.children.length === 8",timeout=30000)
            assert page.locator('#cx-gallery .cx-sample-card img').count()==16
            assert page.locator('#cx-purity').inner_text()!='—'
            assert page.locator('#cx-run').input_value().startswith('v05-')
            page.locator('#cx-layer').select_option('4')
            page.wait_for_function("document.getElementById('cx-gallery')?.children.length === 8",timeout=30000)
            page.locator('#cx-channel').select_option('3')
            page.wait_for_function("document.getElementById('cx-gallery-note')?.textContent.includes('label')",timeout=30000)
            page.locator('#cx-other').select_option('v05-digits-zf-s22')
            page.locator('#cx-compare').click()
            page.wait_for_function("document.getElementById('cx-cka')?.textContent !== '—'",timeout=30000)
            assert page.locator('.cx-match-row').count()>=6
            page.wait_for_function("document.querySelector('#cx-pair-a')?.children.length===6",timeout=30000)
            page.locator('.cx-match-row').nth(2).click()
            page.wait_for_function("document.querySelector('#cx-pair-b')?.children.length===6",timeout=30000)
            page.locator('#cx-sample').fill('2')
            page.locator('#cx-target').select_option('3')
            page.locator('#cx-test').click()
            page.wait_for_function("document.getElementById('cx-auc-gap')?.textContent !== '—'",timeout=30000)
            assert page.locator('#cx-deletion-chart path').count()==2
            page.locator('#concepts').scroll_into_view_if_needed()
            page.locator('#concepts').screenshot(path=str(Path(out)/f'v05-{name}.png'),timeout=30000)
            if width<500:
                scroll=page.evaluate('document.documentElement.scrollWidth')
                assert scroll<=width+2,f'Mobile overflow: scroll={scroll}, viewport={width}'
            assert not errors,errors
            print(name,'passed, CKA=',page.locator('#cx-cka').inner_text(),'gap=',page.locator('#cx-auc-gap').inner_text())
            page.close()
        browser.close()

if __name__=='__main__':run()