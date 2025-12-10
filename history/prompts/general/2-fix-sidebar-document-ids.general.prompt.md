---
id: 2
title: Fix sidebar document IDs mismatch
stage: general
date: 2025-12-10
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: ZC
command: manual fix
labels: [docusaurus, sidebar, documentation, bugfix]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - C:\Users\ZC\Desktop\book\docs\docs\chapters\02-locomotion-systems.md
 - C:\Users\ZC\Desktop\book\docs\docs\chapters\03-perception-systems.md
 - C:\Users\ZC\Desktop\book\docs\sidebars.ts
tests:
---

## Prompt

[ERROR] Loading of version failed for version current

[ERROR] Error: Invalid sidebar file at "sidebars.ts".
These sidebar document ids do not exist:
- chapters/01-introduction-to-physical-ai
- chapters/02-locomotion-systems
- chapters/03-perception-systems

Available document ids are:
- chapters/ch01-introduction-to-physical-ai
- chapters/locomotion-systems
- chapters/perception-systems
- exercises/chapter-01-exercises
- exercises/chapter-02-exercises
- exercises/chapter-03-exercises
- intro
- tutorial-basics/congratulations
- tutorial-basics/create-a-blog-post
- tutorial-basics/create-a-document
- tutorial-basics/create-a-page
- tutorial-basics/deploy-your-site
- tutorial-basics/markdown-features
- tutorial-extras/manage-docs-versions
- tutorial-extras/translate-your-site
- tutorials/chatbot-component-implementation
- tutorials/rag-chatbot-integration-plan

    at Object.checkSidebarsDocIds (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\plugin-content-docs\lib\sidebars\utils.js:245:19)
    at doLoadVersion (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\plugin-content-docs\lib\versions\loadVersion.js:102:19)
    at async loadVersion (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\plugin-content-docs\lib\versions\loadVersion.js:119:16)
    at async Promise.all (index 0)
    at async Object.loadContent (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\plugin-content-docs\lib\index.js:152:33)
    at async C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\core\lib\server\plugins\plugins.js:40:23
    at async Promise.all (index 1)
    at async C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\core\lib\server\plugins\plugins.js:146:25
    at async loadSite (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\core\lib\server\site.js:155:45)
    at async createReloadableSite (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\core\lib\commands\start\utils.js:62:16)

## Response snapshot

I've fixed the sidebar document ID mismatch error by:

1. Adding proper ID fields to the chapter markdown files that were missing them:
   - Added id: ch02-locomotion-systems to 02-locomotion-systems.md
   - Added id: ch03-perception-systems to 03-perception-systems.md
   - Also added proper title, sidebar_label, and description fields for consistency

2. Updated the sidebar configuration in sidebars.ts to use the correct document IDs:
   - Changed 'chapters/01-introduction-to-physical-ai' to 'chapters/ch01-introduction-to-physical-ai'
   - Changed 'chapters/02-locomotion-systems' to 'chapters/ch02-locomotion-systems'
   - Changed 'chapters/03-perception-systems' to 'chapters/ch03-perception-systems'

The mismatch occurred because the sidebar was referencing document IDs that didn't match the actual IDs defined in the markdown files' front matter. The first chapter file had the ID 'ch01-introduction-to-physical-ai', while the second and third chapter files were missing IDs entirely, causing Docusaurus to generate them based on the file path.

## Outcome

- ✅ Impact: Fixed sidebar document ID mismatch that was preventing Docusaurus site build
- 🧪 Tests: No tests required for this fix
- 📁 Files: Modified 3 files to ensure consistent document IDs
- 🔁 Next prompts: Docusaurus site should now build successfully with proper navigation
- 🧠 Reflection: Document IDs in front matter must match sidebar references exactly

## Evaluation notes (flywheel)

- Failure modes observed: Sidebar document ID mismatch causing build failure
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A