# DataWorkshop Research Hub (PWA)

Minimal, accessible, and fast Progressive Web App scaffold for your course project.
Includes:
- GitHub Pages deploy via GitHub Actions
- Lighthouse CI in GitHub Actions (performance/accessibility checks)
- Basic PWA (manifest + service worker + offline fallback)
- Custom domain ready (optional)

## Update the homepage content
Edit `public/index.html` and set the real links for:
- GitHub repo URL
- ED Projects thread URL

## Lighthouse CI
The `lighthouse.yml` job audits the static site from `./public` using Lighthouse CI with thresholds set in `lighthouserc.json`.
You can adjust thresholds as needed.

---

© 2025 Chris. MIT License.
