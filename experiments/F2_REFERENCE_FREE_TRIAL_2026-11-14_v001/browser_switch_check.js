async (page) => {
  await page.getByRole('button', {name:'① 你的原音', exact:true}).click();
  await page.waitForFunction(() => document.getElementById('player').readyState >= 2);
  await page.evaluate(() => { const a=document.getElementById('player');a.pause();a.muted=true;a.currentTime=a.duration-.02; });
  await page.getByRole('button', {name:'② 模型试验版', exact:true}).click();
  await page.waitForFunction(() => document.getElementById('player').readyState >= 2);
  const result=await page.evaluate(() => ({time:document.getElementById('player').currentTime,duration:document.getElementById('player').duration}));
  if(result.time>1) throw new Error('Switch after end failed to return before changes: '+JSON.stringify(result));
  console.log(JSON.stringify(result));
}
