# AshareRank project website

A bilingual, static presentation of the research project. The website is separate from the Python research workflow and does not serve predictions or execute trades.

## Local preview

From the repository root:

```powershell
python -m http.server 4173 --bind 127.0.0.1 --directory website/dist
```

Open `http://127.0.0.1:4173/?lang=zh` or `http://127.0.0.1:4173/?lang=en`. No package installation or build is required. The language preference is stored locally when available; URL language takes precedence. Chinese remains readable without JavaScript.

## Content and publishing

- `dist/index.html`: project content and paired Chinese/English copy.
- `dist/styles.css`: responsive layout, keyboard focus and reduced-motion styles.
- `dist/app.js`: language switching, translated metadata and optional language preference.
- `.openai/hosting.json`: Sites identity and static output directory.

Keep the page aligned with `RESEARCH_PLAN.md`. Do not add unverified performance numbers. Any future empirical result needs a run ID, sample coverage, cost assumptions and uncertainty.

Publish only this website directory through Sites. Use a separate ignored publication checkout so the Sites source repository cannot include research caches, credentials, training outputs or unrelated working changes. The deployed files are the contents of `dist/`.
