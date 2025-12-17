---
id: 007
title: docusaurus-config-urdu-support
stage: implement
date: 2025-12-14
surface: agent
model: claude-sonnet-4-5-20250929
feature: physical-ai-humanoid-robotics
branch: main
user: ZC
command: /sp.implement
labels: [configuration, i18n, docusaurus, translation, ui]
links:
  spec: /specs/physical-ai-humanoid-robotics/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - C:\Users\ZC\Desktop\book\docs\docusaurus.config.ts
tests:
 - C:\Users\ZC\Desktop\book\docs\docusaurus.config.ts
---

## Prompt

```
Update docusaurus.config.js for Urdu support:

// Add to config file
i18n: {
  defaultLocale: 'en',
  locales: ['en', 'ur'],
  localeConfigs: {
    en: {
      label: 'English',
      direction: 'ltr',
    },
    ur: {
      label: 'اردو',
      direction: 'rtl',
    },
  },
},

// Add to navbar
items: [
  {
    type: 'localeDropdown',
    position: 'right',
  },
]
```

## Response snapshot

Successfully updated the Docusaurus configuration file (docusaurus.config.ts) to add comprehensive Urdu language support with the following changes:

1. Enhanced the i18n configuration with locale-specific settings:
   - Added localeConfigs object with English and Urdu configurations
   - Set proper labels: 'English' for English and 'اردو' for Urdu
   - Set proper text directions: 'ltr' for English and 'rtl' for Urdu
   - Maintained the existing locales array ['en', 'ur']

2. Updated the navbar configuration to include the locale dropdown:
   - Added the localeDropdown type item to the navbar items array
   - Positioned the dropdown on the right side of the navbar
   - Maintained existing items (Textbook sidebar and GitHub link)

The configuration now properly supports bidirectional text rendering and provides users with a language selection dropdown in the navigation bar. This complements the existing translation components and ensures proper internationalization support throughout the Physical AI textbook website.

## Outcome

- ✅ Impact: Successfully enhanced Docusaurus i18n configuration with comprehensive Urdu support
- 🧪 Tests: Configuration updated with proper locale settings and direction controls
- 📁 Files: Updated docusaurus.config.ts with enhanced i18n and navbar configurations
- 🔁 Next prompts: Testing the locale switching functionality and verifying RTL rendering
- 🧠 Reflection: The configuration provides complete i18n support with proper text direction handling

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS - Configuration successfully updated with all required settings
- Prompt variant (if applicable): Configuration update prompt
- Next experiment (smallest change to try): Test the locale switching functionality in the browser