// Сбор истории конфигурации ВПП Схипхола с https://en.lvnl.nl/local-community/current-runway-usage-at-schiphol
// Эндпоинт: POST https://en.lvnl.nl/runway/get, тело JSON [год, месяц, день, час, минута] (местное время NL),
// ответ: {runway:{id,start,end (UTC),landing1..3,takeoff1..3,state,isLast}}. curl получает 403 (Cloudflare challenge),
// поэтому запускать в консоли браузера на странице LVNL (Playwright page.evaluate). Шаг 20 мин, пауза 150 мс.
// Сверка с RUNWAY_mvt: scripts/lvnl_check_vs_mvt.py (95,2% движений EHAM на ВПП из объявленного набора, сдвиг 0 ч).
async function lvnlDays(days){ const res={}; for (const [y,m,d] of days){ const seen={};
  for (let mi=0; mi<1440; mi+=20){ const q=[y,m,d,Math.floor(mi/60),mi%60];
    const r=await fetch('/runway/get',{method:'POST',headers:{'content-type':'application/json','accept':'application/json'},body:JSON.stringify(q)});
    const j=await r.json(); const rw=j.runway; if(rw){ rw.queries=(seen[rw.id]?.queries||[]).concat([q.join('-')]); seen[rw.id]=rw; }
    await new Promise(s=>setTimeout(s,150)); }
  res[`${y}-${m}-${d}`]=Object.values(seen); } return res; }
// lvnlDays([[2025,1,15],[2025,7,15],[2026,1,15],[2026,7,15]])
