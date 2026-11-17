async (page) => {
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const versions=await page.locator('[data-version]').evaluateAll(nodes=>nodes.map(n=>n.dataset.version));
  if(JSON.stringify(versions)!==JSON.stringify(['original','model']))throw new Error('Wrong published versions');
  const played=[];
  for(const [key,name] of [['original','① 你的原音'],['model','② 模型试验版']]){
    await page.getByRole('button',{name,exact:true}).click();
    await page.waitForFunction(()=>document.getElementById('player').readyState>=2);
    await page.evaluate(async()=>{const a=document.getElementById('player');a.muted=true;a.currentTime=0;await a.play();});
    await page.waitForFunction(()=>document.getElementById('player').currentTime>.05);
    const state=await page.evaluate(()=>{const a=document.getElementById('player');a.pause();return {src:a.currentSrc,duration:a.duration,error:a.error?.message||null};});
    if(!state.src.endsWith('/audio/'+key+'.wav')||state.error||Math.abs(state.duration-8.612)>.001)throw new Error('Playback failed');
    played.push(state);
  }
  await page.evaluate(()=>{document.getElementById('player').currentTime=3;});
  await page.getByRole('button',{name:'① 你的原音',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('player').readyState>=2);
  const position=await page.locator('#player').evaluate(a=>a.currentTime);
  if(Math.abs(position-3)>.05)throw new Error('Switch lost position');
  const old=await page.request.get('http://127.0.0.1:62512/audio/reference.wav');
  if(old.status()!==404)throw new Error('Withdrawn audio still published');
  await page.screenshot({path:'D:/My_Grand_Projects2026/Voice-trans-Project-2.0/experiments/F2_VERSION_AUDIT_2026-11-14_v001/output/playwright/withdrawn-reference.png',fullPage:true});
  if(errors.length)throw new Error(errors.join('\n'));
  return {versions,played,position_switch:true,removed_reference_http_status:old.status(),script_errors:errors,all_playback_muted:true};
}
