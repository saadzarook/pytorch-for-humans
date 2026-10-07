# Launch checklist (do these by hand)

Everything below needs your accounts, so none of it is automated. Work top to bottom; each step
says what "good" looks like. Commands are for **PowerShell** (Windows); bash versions are in
comments where they differ.

## 0. Before you start

- [ ] Node.js **22.12 or newer** (`node --version`). CI uses Node 24 LTS.
- [ ] Python 3.12+ with the course venv: `.venv\Scripts\activate`.
- [ ] Locally green:
  ```powershell
  python scripts/test_snippets.py          # expect: "28 passed, 0 failed"
  npm ci; npm run check; npm test          # expect: 0 errors; "58 passed"
  npm run build; npm run test:e2e          # expect: "41 passed"
  python scripts/build_notebooks.py --execute   # expect: "... included cells match the website's outputs"
  ```

## 1. GitHub: repo, push, Pages

The repo already exists (`origin` = `github.com/saadzarook/pytorch-for-humans`).

- [ ] Push the new commits: `git push origin main`.
- [ ] **Pages from Actions:** repo → **Settings → Pages → Build and deployment → Source: GitHub
      Actions**. (If the repo is private, Pages needs a paid GitHub plan; make it public otherwise.)
- [ ] Optional, turns on the site's "Open in Kaggle" buttons: **Settings → Secrets and variables →
      Actions → Variables → New repository variable** `KAGGLE_USERNAME` = your Kaggle username.
      (A *variable*, not a secret. It's public on the site anyway.) Re-run "Deploy site" after
      adding it.

> Note: `.mcp.json` and `.vscode/mcp.json` are now untracked, but they're still in the first two
> commits' history. They only contain local paths (your Windows username). Leave them, or rewrite
> history before making the repo public if that matters to you.

## 2. What passing workflow runs look like

Repo → **Actions** tab. Every push to `main` runs three workflows.

### Deploy site (`deploy-site.yml`)
- Job **build**: `Convert notebooks for download` prints `built build/notebooks/…ipynb`;
  `Type-check` ends `0 errors`; `Unit tests (Vitest)` shows `Test Files 5 passed`, `Tests 58 passed`;
  `Build site` ends `11 page(s) built` and `Complete!`; `upload-pages-artifact` succeeds.
- Job **deploy**: green, with the site URL on the run summary:
  `https://saadzarook.github.io/pytorch-for-humans/`.
- Open the URL. Check the lesson page, the **Sim gallery** in the sidebar, and that a sim plays.
  Two known build warnings are harmless: `use astro:head-inject` (upstream Astro) and the
  `i18n` / `404` content notices.

### Test code & notebooks (`notebooks.yml`)
- `Run snippets, lesson facts and sim golden/facts scripts` ends **`28 passed, 0 failed`**, with
  lines like `ok … facts.py claims (12 declared in data.json)`.
- `Convert and execute every notebook` ends **`ran pytorch-for-humans-01-gradient-descent top to
  bottom in …s; included cells match the website's outputs`**.
- Artifact **executed-notebooks** is attached.
- **If it fails only because generated outputs differ** (a last-digit float difference between
  Windows and Linux is possible for the PyTorch outputs): the run attaches
  **regenerated-outputs**. Download it, check the diff is only tiny numeric noise, copy the files
  over yours, run `python scripts/test_snippets.py` locally, and commit.

### Browser tests (`e2e.yml`)
- `Run browser tests` ends **`41 passed`** (takes several minutes; the Pyodide tests download
  Python from its CDN).
- On failure, artifact **playwright-report** has the HTML report and traces. Open
  `playwright-report/index.html` locally to step through the failure.

## 3. First (private) Kaggle push

- [ ] Install the CLI: `pip install -r requirements-kaggle.txt` (kaggle 2.2.4).
- [ ] Authenticate, any way the CLI supports. Check `kaggle --help` first; the docs list
      `kaggle auth login` (browser sign-in), a `KAGGLE_API_TOKEN` environment variable, or
      `~/.kaggle/kaggle.json` from **kaggle.com → Settings → API → Create New Token**.
      Never put credentials in the repo.
- [ ] Build the Kaggle folder. This refuses to run unless both variables are set:
  ```powershell
  $env:KAGGLE_USERNAME = "<your-kaggle-username>"
  $env:COURSE_URL = "https://saadzarook.github.io/pytorch-for-humans/"
  python scripts/build_notebooks.py --kaggle
  # bash: KAGGLE_USERNAME=<you> COURSE_URL=https://… python scripts/build_notebooks.py --kaggle
  ```
- [ ] `python scripts/kaggle_push.py --dry-run`. Expect
      `push  <you>/pytorch-for-humans-01-gradient-descent  (private)` and no "unfilled placeholders".
- [ ] `python scripts/kaggle_push.py`. Kaggle creates the notebook **privately** and runs it.
- [ ] Wait for the run: `kaggle kernels status <you>/pytorch-for-humans-01-gradient-descent`
      until it says complete (a few minutes; CPU only).

### Check the notebook page

Open `https://www.kaggle.com/code/<you>/pytorch-for-humans-01-gradient-descent`. As the owner you
see the same read-only viewer readers will see.

- [ ] **The run succeeded:** the version shows as successful, with no error output in any cell.
- [ ] **Plotly works:**
  - Section 1: the `w = …` **slider** moves the line and updates the loss in the title.
  - Two-knob bowl (section 3): **▶ Play / ⏸ Pause** animate the line and the path.
  - Optimizer race (section 5): **▶ Play** and the **step slider** both work; the three phase
    snapshots below it render.
  - Hovering any chart shows values.
- [ ] **The `to_jshtml` animation plays:** the four-panel "ball rolling downhill" animation
      (section 3) has its own play/step controls and the balls move.
- [ ] **No leftovers:** press Ctrl+F and search the page for `{{`. There should be no matches,
      and no line starting `# include:`.
- [ ] **Facts are filled in:** for example "About **36× steeper across than along**" and
      "**Momentum, then Adam, then SGD** (46, 119 and 280 steps)".
- [ ] **The link back works:** "interactive web version" at the top opens the lesson on your site.
- [ ] Expected, not a bug: the "Bonus (after Copy & Edit)" ipywidgets slider shows nothing
      interactive on the viewer page.

### Make it public (when happy)
- [ ] On Kaggle: the notebook's **Share / Settings → Public**. Or set `"is_private": "false"` in
      `course/foundations/01-gradient-descent/kernel-metadata.json`, rebuild with `--kaggle`, and
      push again.
- [ ] Set the `KAGGLE_USERNAME` repository variable (step 1) if you haven't, and re-run
      "Deploy site", so the lesson's **Open in Kaggle** button appears and points here.

## Things I couldn't verify (check these first if something's off)

- **Kaggle rendering:** the notebook has only been executed locally and in CI, never on Kaggle.
  Plotly picks its Kaggle renderer automatically (it detects `/kaggle/input`); if charts come out
  blank, add `import plotly.io as pio; pio.renderers.default = "kaggle"` to the setup cell.
- **Kaggle CLI auth commands:** the auth options above are from the CLI docs; the exact
  subcommand names can change between CLI versions, so trust `kaggle --help`.
- **First Actions run:** the workflows pass `actionlint` and every step passes locally, but they
  have never run on GitHub. The action versions (`checkout@v7`, `setup-node@v7`, `deploy-pages@v5`,
  …) were the latest releases when written.
