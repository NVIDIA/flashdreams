# FlashDreams Promo Video (HyperFrames)

This folder contains the reproducible HyperFrames project for the FlashDreams promo video.

## Contents

- `index.html` main timeline
- `compositions/` scene compositions
- `assets/styles.css` shared styles
- `assets/flashdreams-logo.svg` logo
- `assets/fonts/` font files
- `package.json` / `package-lock.json` / `hyperframes.json` project config
- `AGENTS.md`, `CLAUDE.md`, `meta.json` project notes and metadata

## Run

Preview:

```bash
npx --yes hyperframes@0.6.54 preview --port 5174 --force-new
```

Render:

```bash
npm run render -- . -c index.html -o renders/flashdreams-promo-hq.mp4 --quality high --resolution landscape-4k
```
