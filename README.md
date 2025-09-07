# DataWorkshop Research Hub (PWA)

Minimal, accessible, and fast Progressive Web App scaffold for your course project.
Includes:
- GitHub Pages deploy via GitHub Actions
- Lighthouse CI in GitHub Actions (performance/accessibility checks)
- Basic PWA (manifest + service worker + offline fallback)
- Custom domain ready (optional)

## Quick start

1. Create a **new GitHub repository** (public) named `dataworkshop-hub` (or any name you prefer).
2. Download this starter and push it to your repo:
   ```bash
   unzip dataworkshop-hub-starter.zip
   cd dataworkshop-hub
   git init && git add . && git commit -m "A03: initial scaffold"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
3. In **Settings → Pages**, choose **GitHub Actions** as the publishing source (Pages will pick up the workflow).
4. Wait for the **Deploy to GitHub Pages** workflow to succeed. Visit the URL shown in the deployment.
5. (Optional) Add a **custom domain** (e.g., `hub.dataworkshop.ai`) in **Settings → Pages** and set DNS:
   - Subdomain (e.g., `hub.dataworkshop.ai`): add a **CNAME** record → `<your-user>.github.io`
   - Apex (`dataworkshop.ai`): add **A** records to GitHub Pages IPs (if you want the root on Pages).

## Update the homepage content
Edit `public/index.html` and set the real links for:
- GitHub repo URL
- ED Projects thread URL

## Lighthouse CI
The `lighthouse.yml` job audits the static site from `./public` using Lighthouse CI with thresholds set in `lighthouserc.json`.
You can adjust thresholds as needed.

## Security
Do **not** commit secrets. If you later deploy to your own SFTP host, store credentials in **GitHub Secrets**.
Rotate any credentials accidentally shared in chats or screenshots.

---

© 2025 Chris. MIT License.
