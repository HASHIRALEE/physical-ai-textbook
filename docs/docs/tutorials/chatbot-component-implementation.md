# Chatbot Component Implementation

## React Chatbot Component

Here's a basic implementation of a chatbot component that could be integrated into the Docusaurus site:

```jsx
import React, { useState, useRef, useEffect } from 'react';
import { useColorMode } from '@docusaurus/theme-common';

const Chatbot = () => {
  const [messages, setMessages] = useState([
    { id: 1, text: 'Hello! I\'m your Physical AI and Humanoid Robotics textbook assistant. Ask me anything about the content!', sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const { colorMode } = useColorMode();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call RAG API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          context: messages.map(m => m.text).join('\n')
        }),
      });

      const data = await response.json();

      // Add bot response
      const botMessage = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot',
        sources: data.sources || []
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const isDarkTheme = colorMode === 'dark';

  return (
    <div className={`chatbot-container ${isDarkTheme ? 'dark' : 'light'}`} style={{
      border: '1px solid #ccc',
      borderRadius: '8px',
      padding: '16px',
      margin: '16px 0',
      backgroundColor: isDarkTheme ? '#2a2a2a' : '#f9f9f9',
      minHeight: '400px',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <div className="chat-messages" style={{
        flex: 1,
        overflowY: 'auto',
        marginBottom: '16px',
        maxHeight: '300px'
      }}>
        {messages.map((message) => (
          <div
            key={message.id}
            style={{
              textAlign: message.sender === 'user' ? 'right' : 'left',
              marginBottom: '12px',
              padding: '8px 12px',
              borderRadius: '8px',
              display: 'inline-block',
              maxWidth: '80%',
              backgroundColor: message.sender === 'user'
                ? (isDarkTheme ? '#0077cc' : '#007bff')
                : (isDarkTheme ? '#444' : '#e9ecef'),
              color: message.sender === 'user' ? 'white' : (isDarkTheme ? 'white' : 'black')
            }}
          >
            {message.text}
            {message.sources && message.sources.length > 0 && (
              <div style={{ fontSize: '0.8em', marginTop: '4px' }}>
                Sources: {message.sources.join(', ')}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div style={{
            textAlign: 'left',
            marginBottom: '12px',
            padding: '8px 12px',
            borderRadius: '8px',
            display: 'inline-block',
            maxWidth: '80%',
            backgroundColor: isDarkTheme ? '#444' : '#e9ecef',
            color: isDarkTheme ? 'white' : 'black'
          }}>
            Thinking...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '8px' }}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a question about Physical AI and Humanoid Robotics..."
          style={{
            flex: 1,
            padding: '8px 12px',
            border: '1px solid #ccc',
            borderRadius: '4px',
            backgroundColor: isDarkTheme ? '#333' : 'white',
            color: isDarkTheme ? 'white' : 'black'
          }}
          disabled={isLoading}
        />
        <button
          type="submit"
          style={{
            padding: '8px 16px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isLoading ? 'not-allowed' : 'pointer'
          }}
          disabled={isLoading || !inputValue.trim()}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default Chatbot;
```

## Integration with Docusaurus

To integrate this component with Docusaurus, you would:

1. Save the component in `src/components/Chatbot.jsx`
2. Import and use it in any page:

```jsx
import Chatbot from '@site/src/components/Chatbot';

// In your MDX file:
<Chatbot />
```

## Backend API Endpoint

Create an API endpoint at `/api/chat` (using Docusaurus' ability to add custom API routes):

```javascript
// In a separate backend service or Docusaurus API routes
app.post('/api/chat', async (req, res) => {
  try {
    const { message, context } = req.body;

    // 1. Retrieve relevant documents from vector database
    const relevantDocs = await retrieveRelevantDocuments(message);

    // 2. Format context for LLM
    const prompt = formatRAGPrompt(message, relevantDocs, context);

    // 3. Generate response using LLM
    const response = await generateLLMResponse(prompt);

    res.json({
      response: response,
      sources: relevantDocs.map(doc => doc.source)
    });
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({
      response: 'Sorry, I encountered an error processing your request.'
    });
  }
});
```

This implementation provides a foundation for RAG chatbot functionality that can be enhanced with more sophisticated features as needed.