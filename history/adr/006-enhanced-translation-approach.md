# ADR 006: Enhanced Translation Component Approach

## Status
Proposed

## Context
The Physical AI and Humanoid Robotics textbook needs to provide sophisticated multilingual support beyond basic translation toggling. The initial TranslateButton component was a good start, but we need a more comprehensive solution that can handle chapter-specific translations with proper content wrapping and advanced RTL styling. The implementation must:
- Support all 16 chapters with specific translations
- Provide better UI/UX for language switching
- Maintain proper content structure during translation
- Integrate seamlessly with Docusaurus-based textbook platform
- Support both English and Urdu content with appropriate styling

## Decision
We will implement an enhanced translation wrapper component with the following approach:
- Create a dedicated UrduTranslationWrapper component that wraps chapter content
- Include comprehensive translation mappings for all 16 chapters
- Implement proper RTL styling and language attributes
- Use state management for translation mode control
- Apply appropriate Urdu fonts ('Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq')
- Ensure responsive design for different screen sizes
- Maintain accessibility standards for both languages

## Alternatives Considered
1. **Server-side Translation**: Pre-translated pages but with increased complexity and maintenance overhead
2. **External Translation Service**: Real-time translation APIs but with potential quality and cost concerns
3. **Simple Toggle Component**: Basic implementation without content wrapping but with reduced functionality
4. **Chapter-specific Components**: Separate translation components for each chapter but with code duplication

## Consequences
### Positive
- Comprehensive translation support for all textbook chapters
- Better user experience with proper content wrapping
- Appropriate RTL styling and typography for Urdu content
- Maintainable code with centralized translation mappings
- Responsive design supporting different devices
- Proper language attributes for accessibility

### Negative
- Increased component complexity compared to basic toggle
- Larger bundle size due to embedded translations
- Need for maintaining translation quality across all chapters
- Potential synchronization issues if chapter content changes

## References
- React component design patterns
- RTL styling best practices
- Urdu typography guidelines
- Accessibility standards for multilingual content
- Docusaurus component integration patterns