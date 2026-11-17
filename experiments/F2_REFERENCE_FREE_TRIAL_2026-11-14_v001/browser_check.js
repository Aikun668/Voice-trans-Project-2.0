async (page) => {
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  if(!await page.getByText('未通过自动效果验证。',{exact:true}).isVisible())throw new Error('Diagnostic status missing');
  const results=[];
  for(const [key,name] of [['original','① 你的原音'],['model','② 模型试验版'],['reference','③ 原唱对应片段']]){
    await page.getByRole('button',{name,exact:true}).click();
    await page.waitForFunction(()=>document.getElementById('player').readyState>=2);
    await page.evaluate(async()=>{const a=document.getElementById('player');a.pause();a.muted=true;a.currentTime=0;await a.play();});
    await page.waitForFunction(()=>document.getElementById('player').currentTime>.03,{timeout:5000});
    const state=await page.evaluate(()=>{const a=document.getElementById('player');a.pause();return{src:a.currentSrc,duration:a.duration,time:a.currentTime,error:a.error?.message||null};});
    if(!state.src.endsWith('/audio/'+key+'.wav')||Math.abs(state.duration-8.612)>.02||state.error)throw new Error('Audio failed: '+JSON.stringify(state));
    results.push({version:key,...state});
  }
  await page.getByRole('button',{name:'从改动处前面听',exact:true}).click();
  if(!await page.getByText('先切回原音或模型版，再定位改动处。',{exact:true}).isVisible())throw new Error('Reference seek notice missing');
  await page.getByRole('button',{name:'① 你的原音',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('player').readyState>=2);
  await page.evaluate(()=>{const a=document.getElementById('player');a.pause();a.currentTime=3.;});
  await page.getByRole('button',{name:'② 模型试验版',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('player').readyState>=2);
  let time=await page.evaluate(()=>document.getElementById('player').currentTime);
  if(Math.abs(time-3)>.05)throw new Error('Version switch lost playback position');
  await page.evaluate(()=>{const a=document.getElementById('player');a.pause();a.currentTime=a.duration-.02;});
  await page.getByRole('button',{name:'① 你的原音',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('player').readyState>=2);
  time=await page.evaluate(()=>document.getElementById('player').currentTime);
  if(time>1)throw new Error('End-of-play switch did not return before correction');
  await page.getByRole('group',{name:'模型版唱得准不准？'}).getByRole('radio',{name:'拿不准',exact:true}).check();
  await page.getByRole('textbox',{name:'哪一句、哪个字，或哪一秒有问题'}).fill('浏览器功能核验，不代表用户反馈。');
  await page.reload();
  if(!await page.getByRole('group',{name:'模型版唱得准不准？'}).getByRole('radio',{name:'拿不准',exact:true}).isChecked())throw new Error('Feedback not retained');
  const downloadPromise=page.waitForEvent('download');
  await page.getByRole('button',{name:'导出反馈',exact:true}).click();
  const download=await downloadPromise;
  await download.saveAs('D:/My_Grand_Projects2026/Voice-trans-Project-2.0/experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/output/playwright/test-feedback.txt');
  await page.getByRole('textbox',{name:'哪一句、哪个字，或哪一秒有问题'}).fill('');
  await page.evaluate(()=>{document.querySelectorAll('input[type=radio]').forEach(e=>e.checked=false);document.getElementById('notes').dispatchEvent(new Event('change',{bubbles:true}));});
  await page.reload();
  if(await page.locator('input:checked').count())throw new Error('Test answers left in the page');
  await page.screenshot({path:'D:/My_Grand_Projects2026/Voice-trans-Project-2.0/experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001/output/playwright/listening-page.png',fullPage:true});
  if(errors.length)throw new Error(JSON.stringify(errors));
  return{audio_versions:results,position_switch:true,end_switch:true,reference_seek_notice:true,feedback_retained:true,feedback_exported:true,test_feedback_cleared:true,script_errors:errors,all_playback_muted:true};
}
