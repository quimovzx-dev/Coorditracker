# Pose Tracker

A browser-based full-body pose tracker. It detects 33 skeletal landmarks (shoulders, elbows, wrists, hips, knees, ankles, and more) and draws them as a dot-and-line skeleton in real time — either from your webcam or from an uploaded photo — and lets you export the tracked coordinates as JSON or CSV.

Everything runs client-side via [MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) (WASM + GPU delegate). No video or image is ever sent to a server.

## Features

- **Live camera tracking** — real-time 33-point pose skeleton overlaid on your webcam feed
- **Upload image tracking** — drop or choose a photo to detect pose landmarks in a static image
- **Adjustable confidence threshold** — filter out low-confidence landmark points
- **Toggleable skeleton lines** — show dots only, or dots + connecting lines
- **Frame logging** (live mode) — record landmark positions over time
- **Data export** — download tracked points as `.json` (structured, with landmark names) or `.csv` (flat, spreadsheet-ready)

## Tech stack

- Vanilla HTML / CSS / JavaScript (no build step, no framework)
- [`@mediapipe/tasks-vision`](https://www.npmjs.com/package/@mediapipe/tasks-vision) — pose landmark detection, loaded from CDN
- Canvas 2D for skeleton rendering

## Running locally

This is a static site — no backend, no build step. Because it uses ES modules and camera access, serve it over HTTP rather than opening the file directly:

```bash
# any static server works, e.g.:
npx serve .
# or
python3 -m http.server 8000
```

Then open the printed local URL in your browser and allow camera access.

## Deploying

Since it's fully static, it deploys as-is to GitHub Pages, Vercel, or Netlify — just point the deploy at the repo root. No environment variables or backend setup required.

## Project structure

```
.
├── index.html        # markup for both live + upload modes
├── css/style.css      # styling
├── js/script.js        # model loading, detection loop, drawing, export
└── README.md
```

## Notes on the pose model

The `pose_landmarker_lite` model is used for speed. For higher accuracy at the cost of performance, swap the `MODEL_URL` in `js/script.js` for `pose_landmarker_full` or `pose_landmarker_heavy` (see [MediaPipe's model list](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models)).
