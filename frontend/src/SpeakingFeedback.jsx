import React from 'react';

const copy = {
  vi: {heading:'Phản hồi phát âm',good:'Phát âm tốt',practice:'Cần luyện thêm',uncertain:'Chưa đủ dữ liệu',heard:'Chưa nghe rõ',phrases:'Từng cụm từ',again:'Nghe mẫu và luyện lại'},
  en: {heading:'Pronunciation feedback',good:'Likely correct',practice:'Practice again',uncertain:'Not enough evidence',heard:'Not heard',phrases:'Phrase feedback',again:'Listen to the model and try again'},
  'zh-TW': {heading:'發音回饋',good:'發音良好',practice:'再練習',uncertain:'證據不足',heard:'未聽清楚',phrases:'詞組回饋',again:'再聽示範並練習'},
  id: {heading:'Umpan balik pelafalan',good:'Pengucapan baik',practice:'Perlu latihan',uncertain:'Bukti belum cukup',heard:'Tidak terdengar',phrases:'Umpan balik frasa',again:'Dengarkan contoh dan coba lagi'},
  hi: {heading:'उच्चारण प्रतिक्रिया',good:'उच्चारण अच्छा है',practice:'फिर अभ्यास करें',uncertain:'पर्याप्त प्रमाण नहीं',heard:'सुनाई नहीं दिया',phrases:'वाक्यांश प्रतिक्रिया',again:'नमूना सुनें और फिर कोशिश करें'}
};

export default function SpeakingFeedback({result,locale='vi'}) {
  if (!result) return null;
  const t=copy[locale]||copy.vi;
  const summary=result.summary||{};
  const phrases=Array.isArray(result.phrases)?result.phrases:[];
  const count=key=>Number(summary[key]||0);
  const uncertain=count('insufficient_evidence')+count('uncertain')+count('unassessable');
  const statuses={likely_correct:t.good,practice:t.practice,insufficient_evidence:t.uncertain,uncertain:t.uncertain,unassessable:t.uncertain,not_heard:t.heard};
  return <section className="feedback panel" aria-live="polite">
    <h3>{t.heading}</h3>
    <div className="feedback-counts">
      <span className="feedback-count good"><strong>{count('likely_correct')}</strong>{t.good}</span>
      <span className="feedback-count needs-practice"><strong>{count('practice')}</strong>{t.practice}</span>
      <span className="feedback-count unsure"><strong>{uncertain}</strong>{t.uncertain}</span>
    </div>
    {phrases.length>0&&<><h4>{t.phrases}</h4><ul className="phrase-feedback">{phrases.map((phrase,index)=><li key={index} className={phrase.status||'uncertain'}><span className="chinese">{phrase.annotation||phrase.word||phrase.text||`#${index+1}`}</span><span>{statuses[phrase.status]||t.uncertain}</span></li>)}</ul></>}
    {count('practice')+uncertain>0&&<p className="muted">{t.again}</p>}
  </section>;
}
