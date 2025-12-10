# 1. Textbook Architecture Decisions

Date: 2025-12-10

## Status

Accepted

## Context

We need to create a comprehensive, AI-native textbook for a university-level course on Physical AI and Humanoid Robotics. The textbook must be structured in 2000-word sections, include learning objectives, theoretical foundations, practical examples, code snippets (Python/ROS2), interactive exercises, and summaries. It must use Docusaurus structure with GitHub Pages deployment and integrated RAG chatbot.

## Decision

We will use the following architecture:

1. **Platform**: Docusaurus as the static site generator
2. **Content Structure**: Markdown files organized in chapters, tutorials, and exercises
3. **Deployment**: GitHub Pages via GitHub Actions workflow
4. **Technology Stack**:
   - Frontend: Docusaurus/React
   - Backend: Node.js for RAG service (planned)
   - Database: Vector database for RAG (planned)
   - Languages: Python/ROS2 for code examples
5. **Development Methodology**: Spec-Kit Plus with feature specifications, plans, and tasks

## Rationale

- **Docusaurus**: Provides excellent documentation features, search, and theming capabilities
- **Markdown**: Simple, version-controllable content format that's familiar to educators
- **GitHub Pages**: Cost-effective, reliable hosting with good integration with GitHub workflows
- **Modular Structure**: Allows for incremental development and maintenance
- **Spec-Kit Plus**: Ensures systematic development with clear requirements and validation

## Consequences

### Positive
- Easy content management and version control
- Scalable architecture that can grow with additional chapters
- Cost-effective deployment solution
- Familiar tools for potential contributors
- Good SEO and accessibility features

### Negative
- Learning curve for team members unfamiliar with Docusaurus
- RAG implementation will require additional infrastructure
- Static content updates require rebuild and redeployment

## Alternatives Considered

1. **Traditional LMS Platform**: More complex but with built-in assessment features
2. **Jupyter Books**: Good for interactive content but less flexible for complex layouts
3. **Custom React Application**: More control but more development overhead

## Notes

This architecture supports the core requirements of university-level content, interactive elements, and planned RAG chatbot integration while maintaining simplicity for content authors.