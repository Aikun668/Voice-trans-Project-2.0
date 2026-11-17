async (page) => {
  const errors=[];const downloads=[];
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('download',d=>downloads.push(d.suggestedFilename()));
  await page.addInitScript(()=>{
    window.__copied='';window.__blocked=false;
    Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{
      if(window.__blocked)throw new Error('Blocked for fallback check');
      window.__copied=text;
    }}});
  });
  await page.reload();
  await page.waitForFunction(()=>document.querySelector('#player').readyState>=2);
  const playback=[];
  for(const version of ['original','model']){
    await page.locator('[data-version="'+version+'"]').click();
    await page.waitForFunction(v=>document.querySelector('#player').currentSrc.endsWith('/audio/'+v+'.wav')&&document.querySelector('#player').readyState>=2,version);
    await page.evaluate(()=>{const p=document.querySelector('#player');p.muted=true;p.currentTime=0;});
    await page.getByRole('button',{name:'从头听',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('#player').currentTime>.15);
    playback.push(await page.evaluate(()=>{const p=document.querySelector('#player');p.pause();return {url:p.currentSrc,duration:p.duration,advanced:p.currentTime};}));
  }
  await page.evaluate(()=>{document.querySelector('#player').currentTime=2.72;});
  await page.getByRole('button',{name:'① 原音 · 同样响',exact:true}).click();
  await page.waitForFunction(()=>Math.abs(document.querySelector('#player').currentTime-2.72)<.08&&document.querySelector('#player').readyState>=2);
  await page.getByRole('button',{name:'听第二处改动',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#player').currentTime>=4.55);
  await page.evaluate(()=>{const p=document.querySelector('#player');p.pause();p.currentTime=p.duration-.01;p.playbackRate=4;return p.play();});
  await page.waitForFunction(()=>document.querySelector('#player').ended);
  await page.getByRole('button',{name:'② MusicBERT 原权重适配版',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#player').readyState>=2&&document.querySelector('#player').currentTime<.1);
  await page.evaluate(()=>{document.querySelector('#player').playbackRate=1;});
  await page.getByRole('group',{name:'处理版唱得准不准？'}).getByRole('radio',{name:'拿不准',exact:true}).check();
  await page.getByRole('group',{name:'声音有没有新问题？'}).getByRole('radio',{name:'拿不准',exact:true}).check();
  await page.getByRole('group',{name:'你会选哪个？'}).getByRole('radio',{name:'拿不准',exact:true}).check();
  await page.getByRole('textbox',{name:'哪一句、哪个字，或哪一秒有问题'}).fill('浏览器功能检查，不代表 Jade 试听反馈。');
  await page.getByRole('button',{name:'复制反馈',exact:true}).click();
  await page.getByText('已复制，直接粘贴到聊天里即可。',{exact:true}).waitFor();
  const copied=await page.evaluate(()=>window.__copied);
  if(!copied.includes('\n音准：拿不准\n新问题：拿不准\n选择：拿不准\n'))throw new Error('Copied values/newlines wrong');
  if(!copied.includes('f5bb685afe128bc9e0e445be51b5333b98be6a5456df850bd9c9f2cc80bb7275'))throw new Error('Wrong model identity');
  await page.evaluate(()=>window.__blocked=true);
  await page.getByRole('button',{name:'复制反馈',exact:true}).click();
  const fallback=await page.evaluate(()=>{const e=document.querySelector('#feedback-text');return {visible:!e.hidden,selected:e.selectionStart===0&&e.selectionEnd===e.value.length,value:e.value};});
  if(!fallback.visible||!fallback.selected||fallback.value!==copied)throw new Error('Clipboard fallback failed');
  await page.reload();
  if((await page.getByRole('textbox',{name:'哪一句、哪个字，或哪一秒有问题'}).inputValue())!=='浏览器功能检查，不代表 Jade 试听反馈。')throw new Error('Feedback persistence failed');
  const range=await page.request.get(new URL('audio/model.wav',page.url()).href,{headers:{Range:'bytes=0-159'}});
  if(range.status()!==206||(await range.body()).length!==160)throw new Error('WAV range response failed');
  if(downloads.length)throw new Error('Feedback unexpectedly downloaded');
  await page.evaluate(()=>{localStorage.removeItem('musicbert-frozen-20261114-v001');});
  await page.reload();
  await page.setViewportSize({width:1100,height:1000});
  await page.screenshot({path:'output/playwright/listening-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
  if(overflow)throw new Error('Mobile horizontal overflow');
  await page.screenshot({path:'output/playwright/listening-mobile.png',fullPage:true});
  await page.setViewportSize({width:1100,height:1000});
  if(errors.length)throw new Error('Page errors: '+errors.join('; '));
  return {playback,position_preserved:true,end_switch_restarts:true,seek_buttons:true,
          copy_success:true,copy_fallback_selected:true,feedback_persisted:true,
          feedback_test_cleared:true,feedback_downloads:downloads,range_status:range.status(),
          mobile_overflow:false,page_errors:errors,clipboard_api_simulated:true,system_clipboard_not_touched:true};
}
