---
id: 1
title: Fix YAML front matter error
stage: general
date: 2025-12-10
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: ZC
command: manual fix
labels: [yaml, docusaurus, bugfix, documentation]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - C:\Users\ZC\Desktop\book\docs\docs\chapters\01-introduction-to-physical-ai.md
tests:
---

## Prompt

[ERROR] Error while parsing Markdown front matter.
This can happen if you use special characters in front matter values (try using double quotes around that value).
[ERROR] Loading of version failed for version current

[ERROR] Error: Can't process doc metadata for doc at path path=C:\Users\ZC\Desktop\book\docs\docs\chapters\01-introduction-to-physical-ai.md in version name=current
    at processDocMetadata (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\plugin-content-docs\lib\docs.js:152:15)
    at async Promise.all (index 1)
    at async loadVersionDocsBase (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\plugin-content-docs\lib\versions\loadVersion.js:63:18)
    at async doLoadVersion (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\plugin-content-docs\lib\versions\loadVersion.js:72:22)
    at async loadVersion (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\plugin-content-docs\lib\versions\loadVersion.js:119:16)
    at async Promise.all (index 0)
    at async Object.loadContent (C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\plugin-content-docs\lib\index.js:152:33)
    at async C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\core\lib\server\plugins\plugins.js:40:23
    at async Promise.all (index 1)
    at async C:\Users\ZC\Desktop\book\docs\node_modules\@docusaurus\core\lib\server\plugins\plugins.js:146:25 {
  [cause]: YAMLException: incomplete explicit mapping pair; a key node is missed; or followed by a non-tabulated empty line at line 4, column 25:
      sidebar_label: Chapter 1: Introduction
                              ^
      at generateError (C:\Users\ZC\Desktop\book\docs\node_modules\gray-matter\node_modules\js-yaml\lib\js-yaml\loader.js:183:10)
      at throwError (C:\Users\ZC\Desktop\book\docs\node_modules\gray-matter\node_modules\js-yaml\lib\js-yaml\loader.js:189:9)
      at readBlockMapping (C:\Users\ZC\Desktop\book\docs\node_modules\gray-matter\node_modules\js-yaml\lib\js-yaml\loader.js:1043:9)
      at composeNode (C:\Users\ZC\Desktop\book\docs\node_modules\gray-matter\node_modules\js-yaml\lib\js-yaml\loader.js:1375:12)
      at readDocument (C:\Users\ZC\Desktop\book\docs\node_modules\gray-matter\node_modules\js-yaml\lib\js-yaml\loader.js:1541:3)
      at loadDocuments (C:\Users\ZC\Desktop\book\docs\node_modules\gray-matter\node_modules\js-yaml\lib\js-yaml\loader.js:1604:5)
      at load (C:\Users\ZC\Desktop\book\docs\node_modules\gray-matter\node_modules\js-yaml\lib\js-yaml\loader.js:1630:19)
      at Object.safeLoad (C:\Users\ZC\Desktop\book\docs\node_modules\gray-matter\node_modules\js-yaml\lib\js-yaml\loader.js:1653:10)
      at module.exports (C:\Users\ZC\Desktop\book\docs\node_modules\gray-matter\lib\parse.js:12:17)
      at parseMatter (C:\Users\ZC\Desktop\book\docs\node_modules\gray-matter\index.js:109:17) {
    reason: 'incomplete explicit mapping pair; a key node is missed; or followed by a non-tabulated empty line',
    mark: Mark {
      name: null,
      buffer: '\n' +
        'id: ch01-introduction-to-physical-ai\n' +
        'title: Introduction to Physical AI\n' +
        'sidebar_label: Chapter 1: Introduction\n' +
        'sidebar_position: 1\n' +
        'description: Introduction to the fundamentals of Physical AI and embodied intelligence\n' +
        '\x00',
      position: 97,
      line: 3,
      column: 24
    }
  }
} solve th erroe

## Response snapshot

I'll help you fix the YAML front matter error in the markdown file. The error indicates there's an issue with the YAML syntax in the file C:\Users\ZC\Desktop\book\docs\docs\chapters\01-introduction-to-physical-ai.md.

The issue is in the YAML front matter where `sidebar_label: Chapter 1: Introduction` contains a colon followed by a space, which the YAML parser interprets as a key-value separator. This makes it think "Introduction" is a separate key-value pair, causing the parsing error.

I've fixed this by properly quoting the value containing the colon:

```yaml
sidebar_label: "Chapter 1: Introduction"
```

This change ensures the entire string "Chapter 1: Introduction" is treated as a single value, resolving the YAML parsing error and allowing Docusaurus to build successfully.

## Outcome

- ✅ Impact: Fixed YAML parsing error that was preventing Docusaurus site build
- 🧪 Tests: No tests required for this fix
- 📁 Files: Modified docs\docs\chapters\01-introduction-to-physical-ai.md
- 🔁 Next prompts: Docusaurus site should now build successfully
- 🧠 Reflection: Proper YAML quoting is essential when values contain colons followed by spaces

## Evaluation notes (flywheel)

- Failure modes observed: YAML parsing error due to unquoted values containing colons
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A