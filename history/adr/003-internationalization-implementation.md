# ADR 003: Internationalization Implementation for Physical AI Textbook

## Status
Proposed

## Context
The Physical AI and Humanoid Robotics textbook needs to support multiple languages to accommodate diverse student populations. The initial requirement specified English as the primary language with Urdu translation as an option. The implementation must:
- Support right-to-left (RTL) text rendering for Urdu
- Provide appropriate font families for Urdu content
- Maintain content integrity when switching between languages
- Integrate seamlessly with the Docusaurus-based textbook platform

## Decision
We will implement internationalization support with the following approach:
- Use Docusaurus built-in i18n capabilities with 'en' and 'ur' locales
- Integrate Google Fonts Noto Nastaliq Urdu for proper text rendering
- Implement a React-based translation toggle component (TranslateButton)
- Apply RTL styling and appropriate font families dynamically
- Maintain content in both languages with preservation of original formatting

## Alternatives Considered
1. **Machine Translation API**: Real-time translation services but with potential accuracy issues for technical content
2. **Static Bilingual Pages**: Pre-translated pages but with maintenance complexity
3. **External Translation Service**: Third-party services but with dependency and cost concerns
4. **Simple Toggle Component**: Basic implementation without proper RTL support

## Consequences
### Positive
- Students can access content in their preferred language
- Proper RTL text rendering and styling for Urdu content
- Dynamic switching between languages without page reload
- Maintains the educational quality of technical content in both languages
- Scalable approach for adding additional languages in the future

### Negative
- Additional complexity in content management
- Increased page load times due to font loading
- Need for content maintenance in multiple languages
- Potential for translation inconsistencies

## References
- Docusaurus i18n documentation
- Google Fonts Noto Nastaliq Urdu specification
- RTL CSS best practices
- React state management patterns