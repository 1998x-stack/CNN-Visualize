"""Playwright smoke test with live FastAPI responses via browser routing.

Uses the real TestClient app as a transport to avoid infrastructure restrictions
on localhost access from Chromium; no mocked visualization payloads.
"""
from __future__ import annotations
from pathlib import Path
import re
from urllib.parse import urlparse
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
from cnnviz.server import app


def run(out='artifacts/browser'):
    output=Path(out)
    output.mkdir(parents=True,exist_ok=True)
    client=TestClient(app)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True, executable_path='/usr/bin/chromium', args=['--no-sandbox'])
        for label,w,h in [('desktop',1440,900),('mobile',390,844)]:
            page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
            errors=[]
            page.on('pageerror',lambda err: errors.append(str(err)))
            def transport(source,request_url):
                u=urlparse(request_url)
                r=client.get(u.path+('?' + u.query if u.query else ''))
                return {'status':r.status_code,'body':r.text,'content_type':r.headers.get('content-type','application/json')}
            page.expose_binding('backendFetch',transport)
            page.evaluate("window.fetch=async function(url){const r=await window.backendFetch(String(url));return new Response(r.body,{status:r.status,headers:{'content-type':r.content_type}})}")
            source=Path('web/index.html').read_text(encoding='utf-8')
            source=re.sub(r'<script src="/(?:app|dynamics).js" defer></script>','',source)
            source=re.sub(r'<link rel="stylesheet" href="/(?:app|dynamics).css"\s*/>','',source)
            page.set_content(source,wait_until='domcontentloaded')
            page.add_style_tag(content=Path('web/app.css').read_text())
            page.add_style_tag(content=Path('web/dynamics.css').read_text())
            page.add_script_tag(content=Path('web/app.js').read_text())
            page.add_script_tag(content=Path('web/dynamics.js').read_text())
            page.wait_for_function("document.querySelector('#dyn-message')?.textContent?.includes('Real checkpoints')",timeout=25000)
            assert int(page.locator('#dyn-run-count').inner_text())>=2
            assert page.locator('.dyn-cka-cell').count()>100
            assert page.locator('#dyn-image-diff').get_attribute('src').startswith('data:image/png;base64,')
            page.locator('#dyn-layer').select_option('5')
            page.wait_for_function("document.querySelector('#dyn-channel-label')?.textContent?.includes('CH') && document.querySelector('#dyn-cka-score')?.textContent !== '—'",timeout=25000)
            page.locator('.dyn-channel').nth(3).click()
            page.wait_for_function("document.querySelector('#dyn-channel-label').textContent.includes('CH 03')",timeout=10000)
            page.locator('#dyn-step-b').select_option('128')
            page.wait_for_function("document.querySelector('#dyn-pair-label').textContent.includes('128')",timeout=15000)
            page.locator('#dyn-sample').fill('7')
            page.wait_for_function("document.querySelector('#dyn-sample-label').textContent.includes('07')",timeout=10000)
            page.locator('#dyn-run').select_option('research-digits-zf-v04')
            page.wait_for_function("document.querySelector('#dyn-arch')?.textContent.includes('zf-small')",timeout=25000)
            page.wait_for_function("document.querySelector('#dyn-message')?.textContent.includes('Real checkpoints')",timeout=25000)
            page.locator('#dynamics').scroll_into_view_if_needed()
            page.locator('#dynamics').screenshot(path=str(output/f'v04-{label}.png'),timeout=20000)
            if w<500:
                total=page.evaluate('document.documentElement.scrollWidth')
                assert total<=w+2,f'mobile horizontal overflow {total} > {w}'
            assert not errors,errors
            print(label,'browser passed; screenshot',output/f'v04-{label}.png')
            page.close()
        browser.close()


if __name__=='__main__':run()