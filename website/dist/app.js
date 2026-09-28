(() => {
  const buttons = document.querySelectorAll('[data-lang]');
  const descriptions = {
    zh: 'AshareRank：研究 A 股五日超额收益的横截面排序，关注可复现的实验与成本后的表现。',
    en: 'AshareRank studies five-day excess-return ranking in China A-shares through reproducible experiments and cost-adjusted evaluation.'
  };
  function setLanguage(language) {
    const lang = language === 'en' ? 'en' : 'zh';
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-zh][data-en]').forEach(element => {
      element.textContent = element.dataset[lang].replace(/\\n/g, '\n');
    });
    document.querySelectorAll('[data-aria-zh]').forEach(element => {
      element.setAttribute('aria-label', element.getAttribute(`data-aria-${lang}`));
    });
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.lang === lang)));
    document.title = lang === 'zh' ? 'AshareRank — A 股横截面排序研究' : 'AshareRank — Cross-sectional ranking research';
    document.querySelector('meta[name="description"]').content = descriptions[lang];
    try { localStorage.setItem('asharerank-language', lang); } catch { /* Storage is optional. */ }
    try {
      const url = new URL(location.href);
      url.searchParams.set('lang', lang);
      history.replaceState(null, '', url);
    } catch { /* Also works when opened directly as a local file. */ }
  }
  let saved;
  try { saved = localStorage.getItem('asharerank-language'); } catch { /* Use browser preference. */ }
  const requested = new URLSearchParams(location.search).get('lang');
  const preferred = navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en';
  setLanguage(['zh', 'en'].includes(requested) ? requested : ['zh', 'en'].includes(saved) ? saved : preferred);
  buttons.forEach(button => button.addEventListener('click', () => setLanguage(button.dataset.lang)));
})();
