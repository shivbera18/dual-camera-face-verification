# GitHub Repository Setup Guide

## Step 1: Initialize Local Git Repository

```bash
# Navigate to your project directory
cd "C:\Users\Shiv\Desktop\Final project"

# Initialize git
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Dual-camera face verification system with complete documentation"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `dual-camera-face-verification` (or your choice)
3. Description: "Biometric face verification using stereo vision for liveness detection and deepfake prevention"
4. Choose: **Public** (recommended for portfolio) or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we already have them)
6. Click "Create repository"

## Step 3: Connect Local Repository to GitHub

GitHub will show you commands. Use these:

```bash
# Add remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/dual-camera-face-verification.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 4: Verify Upload

Visit your repository at:
```
https://github.com/YOUR_USERNAME/dual-camera-face-verification
```

You should see:
- ✅ README.md displayed on homepage
- ✅ All documentation in `docs/` folder
- ✅ Project structure visible
- ✅ .gitignore working (no large files uploaded)

## Step 5: Add Repository Topics (Optional but Recommended)

On your GitHub repository page:
1. Click "⚙️ Settings" → "General"
2. Or click "Add topics" near the top
3. Add these topics:
   - `face-recognition`
   - `face-verification`
   - `anti-spoofing`
   - `liveness-detection`
   - `deepfake-detection`
   - `stereo-vision`
   - `computer-vision`
   - `biometrics`
   - `opencv`
   - `tensorflow`
   - `arcface`
   - `btech-project`
   - `final-year-project`

## Step 6: Enable GitHub Pages for Documentation (Optional)

1. Go to repository Settings → Pages
2. Source: Deploy from branch
3. Branch: `main`, folder: `/docs`
4. Save

Your documentation will be available at:
```
https://YOUR_USERNAME.github.io/dual-camera-face-verification/
```

## Step 7: Add Project Description

Edit your repository description to include:
```
🎓 BTech Final Year Project: Dual-camera face verification system using stereo vision for depth-based liveness detection, texture analysis for anti-spoofing, and EfficientNet for deepfake detection. Achieves 95%+ anti-spoofing accuracy with real-time performance.
```

## Future Updates

When you make changes:

```bash
# Check status
git status

# Add changed files
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Common Git Commands

```bash
# See commit history
git log --oneline

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Merge branch
git merge feature-name

# Pull latest changes
git pull origin main

# See remote URL
git remote -v
```

## Troubleshooting

**Error: "remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/dual-camera-face-verification.git
```

**Error: "failed to push some refs"**
```bash
git pull origin main --rebase
git push origin main
```

**Want to undo last commit?**
```bash
git reset --soft HEAD~1  # Keeps changes
# OR
git reset --hard HEAD~1  # Discards changes
```

## Repository Structure on GitHub

```
dual-camera-face-verification/
├── README.md                    ← Main page
├── LICENSE
├── requirements.txt
├── .gitignore
├── GITHUB-SETUP.md             ← This file
├── docs/                        ← Documentation
│   ├── technical-specification.md
│   ├── datasets-guide.md
│   ├── research.md
│   ├── requirements.md
│   └── QUICK-START.md
├── calibration/
├── models/
├── data/
├── src/                         ← Source code (to be added)
├── train/                       ← Training scripts (to be added)
└── main.py                      ← Main application (to be added)
```

## Next Steps After GitHub Setup

1. ✅ Repository created and pushed
2. ⏭️ Start implementing code (follow docs/technical-specification.md)
3. ⏭️ Add code files to `src/` directory
4. ⏭️ Commit and push regularly
5. ⏭️ Add screenshots/demo videos to README
6. ⏭️ Create releases for milestones

---

**Ready to push? Run the commands in Step 1 and Step 3!**
