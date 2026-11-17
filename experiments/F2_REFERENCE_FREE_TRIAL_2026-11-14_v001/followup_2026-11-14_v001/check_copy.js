async (page) => {
  await page.addInitScript(() => {
    window.__copyText='';window.__copyBlocked=false;
    Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{
      if(window.__copyBlocked)throw new Error('Clipboard blocked for test');
      window.__copyText=text;
    }}});
  });
  await page.reload();
  const downloads=[];page.on('download',d=>downloads.push(d.suggestedFilename()));
  await page.getByRole('group',{name:'模型版唱得准不准？'}).getByRole('radio',{name:'差不多',exact:true}).check();
  await page.getByRole('group',{name:'声音有没有新问题？'}).getByRole('radio',{name:'没听到新问题',exact:true}).check();
  await page.getByRole('group',{name:'你会选哪个？'}).getByRole('radio',{name:'都不行',exact:true}).check();
  await page.getByRole('textbox',{name:'哪一句、哪个字，或哪一秒有问题'}).fill('复制功能核验，不代表用户反馈。');
  await page.locator('#export').click();
  await page.waitForTimeout(150);
  const text=await page.evaluate(()=>window.__copyText);
  if(downloads.length)throw new Error('Feedback unexpectedly downloaded: '+downloads.join(','));
  if(!text.includes('\n音准：差不多\n新问题：没听到新问题\n选择：都不行\n'))throw new Error('Copied feedback missing actual form values or newlines');
  if(!text.includes('d63eea5b0dea900aa4fb100fbb0159c0e4cba3177821d24ac6e20ceebdca955d'))throw new Error('Feedback model identity wrong');
  if(!await page.getByText('已复制，直接粘贴到聊天里即可。',{exact:true}).isVisible())throw new Error('Copy success message missing');
  await page.evaluate(()=>window.__copyBlocked=true);
  await page.locator('#export').click();
  await page.waitForTimeout(150);
  const fallback=await page.evaluate(()=>{
    const e=document.getElementById('feedback-text');
    return e?{visible:!e.hidden,selected:e.selectionStart===0&&e.selectionEnd===e.value.length,text:e.value}:null;
  });
  if(!fallback?.visible||!fallback.selected||fallback.text!==text)throw new Error('Clipboard fallback did not show/select the same text');
  if(downloads.length)throw new Error('Fallback triggered download');
  if(await page.locator('[data-version="reference"]').count())throw new Error('Withdrawn reference was restored');
  await page.getByRole('textbox',{name:'哪一句、哪个字，或哪一秒有问题'}).fill('');
  await page.evaluate(()=>{document.querySelectorAll('input[type=radio]').forEach(e=>e.checked=false);document.getElementById('notes').dispatchEvent(new Event('change',{bubbles:true}));});
  await page.reload();
  return{copied_feedback:text,copy_success:true,fallback_selected:true,feedback_downloads:downloads,reference_stays_withdrawn:true,system_clipboard_not_touched:true};
}
