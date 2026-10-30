"""Optional Chromium smoke test: real FastAPI results delivered into a sandboxed browser.

Requires Playwright Python and installed Chromium; deliberately uses no external network.
Run: python tests/browser_smoke.py
"""
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
from playwright.async_api import async_playwright
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from cnnviz.server import app
client=TestClient(app)
html=(root/'web/index.html').read_text().replace('<link rel="stylesheet" href="/app.css" />','').replace('<script src="/app.js" defer></script>','')
css=(root/'web/app.css').read_text();js=(root/'web/app.js').read_text()
async def build_page(browser,w,h):
 page=await browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 async def backend(_, url):
  if url.startswith('/api/trajectory') and 'sample_id=695' in url:
   await asyncio.sleep(.35)
  resp=await asyncio.to_thread(client.get,url)
  return {'ok':resp.is_success,'status':resp.status_code,'data':resp.json()}
 await page.expose_binding('backendFetch',backend)
 await page.set_content(html,wait_until='domcontentloaded')
 await page.add_style_tag(content=css)
 await page.evaluate("""() => { window.fetch=(url,opts={}) => {
    const signal=opts.signal;
    const aborted=()=>new DOMException('Request aborted','AbortError');
    if(signal?.aborted)return Promise.reject(aborted());
    return window.backendFetch(url).then(result=>{
        if(signal?.aborted)throw aborted();
        return {ok:result.ok,status:result.status,json:async()=>result.data};
    });
  }}""")
 await page.add_script_tag(content=js)
 return page,errors

async def main():
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
  page,errors=await build_page(browser,1440,900)
  await page.wait_for_function("document.querySelector('#compare-rms')?.textContent !== '—' && document.querySelector('#compare-rms')?.textContent !== '…'",timeout=15000)
  print('initial',await page.locator('#run-title').inner_text(),'target',await page.locator('#target-class').input_value(),'rms',await page.locator('#compare-rms').inner_text())
  await page.locator('#epoch-slider').fill('7');await page.locator('#epoch-slider').dispatch_event('input');
  await page.locator('#target-class').select_option('8')
  await page.wait_for_function("document.querySelector('#epoch-insight').textContent.includes('固定目标「8」')",timeout=15000)
  await page.locator('#epoch-a').select_option('3');await page.locator('#epoch-b').select_option('7')
  await page.wait_for_function("document.querySelector('#epoch-a-label').textContent.includes('03') && document.querySelector('#epoch-b-label').textContent.includes('07')",timeout=15000)
  await page.locator('#scale-mode').select_option('local')
  await page.wait_for_function("document.querySelector('#compare-note').textContent.includes('各自拉伸')",timeout=15000)
  await page.locator('[data-layer="3"]').click()
  await page.wait_for_function("document.querySelector('#channel-count').textContent.includes('28')",timeout=15000)
  await page.locator('.sample').nth(3).click();await page.locator('.sample').nth(0).click()
  await page.wait_for_function("document.querySelector('#compare-rms').textContent !== '…'",timeout=15000)
  await page.wait_for_function('state.info?.sample_id === state.sample?.id && state.compareInfo?.sample_id === state.sample?.id',timeout=15000)
  await page.wait_for_timeout(450)
  consistent=await page.evaluate("({sample:state.sample.id,inspect:state.info.sample_id,compare:state.compareInfo.sample_id,layer:state.layer,reportedLayer:state.info.layer})")
  assert consistent['sample']==consistent['inspect']==consistent['compare']
  assert consistent['layer']==consistent['reportedLayer']
  print('sample',await page.locator('.sample.active').count(),'pred',await page.locator('#predicted-label').inner_text(),'compare',await page.locator('#compare-rms').inner_text(),'errors',errors)
  await page.screenshot(path=str(root/'preview-v022-desktop.png'),full_page=True)
  assert not errors, errors
  mobile,mobile_errors=await build_page(browser,390,844)
  await mobile.wait_for_function("document.querySelector('#compare-rms')?.textContent !== '—' && document.querySelector('#compare-rms')?.textContent !== '…'",timeout=15000)
  await mobile.screenshot(path=str(root/'preview-v022-mobile.png'),full_page=True)
  scroll=await mobile.evaluate('document.documentElement.scrollWidth')
  print('mobile scroll',scroll,'viewport',await mobile.evaluate('innerWidth'),'errors',mobile_errors)
  assert scroll<=392, f'mobile horizontal overflow: {scroll}'
  assert not mobile_errors,mobile_errors
  await browser.close()
asyncio.run(main())