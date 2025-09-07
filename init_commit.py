# Create a minimal PWA starter + CI/CD workflows and a 1‑page PDF deliverable
import os, json, textwrap, shutil
from datetime import datetime

root = "/mnt/data/dataworkshop-hub"
public = os.path.join(root, "public")
workflows = os.path.join(root, ".github", "workflows")
os.makedirs(public, exist_ok=True)
os.makedirs(workflows, exist_ok=True)

# ---------- LICENSE ----------
license_text = """MIT License

Copyright (c) 2025 Chris

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
with open(os.path.join(root, "LICENSE"), "w") as f:
    f.write(license_text)

# ---------- README ----------
readme = f"""# DataWorkshop Research Hub (PWA)

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
   git init && git add . && git commit -m "initial scaffold"
   git branch -M main
   git remote add origin https://github.com/cesposo/thedataworkshop.git
   git push -u origin main
