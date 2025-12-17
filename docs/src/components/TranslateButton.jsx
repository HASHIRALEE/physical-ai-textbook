import React, { useState } from 'react';

const TranslateButton = ({ content }) => {
  const [isUrdu, setIsUrdu] = useState(false);
  const [originalContent, setOriginalContent] = useState(content);

  const toggleTranslation = async () => {
    if (!isUrdu) {
      // Translate to Urdu (demo for hackathon)
      const urduContent = `اردو ترجمہ: ${content.substring(0, 200)}...`;
      setOriginalContent(content);
      document.querySelector('.chapter-content').innerHTML = urduContent;
      document.body.style.fontFamily = "'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq'";
      document.body.style.direction = 'rtl';
    } else {
      // Back to English
      document.querySelector('.chapter-content').innerHTML = originalContent;
      document.body.style.fontFamily = 'inherit';
      document.body.style.direction = 'ltr';
    }
    setIsUrdu(!isUrdu);
  };

  return (
    <button
      onClick={toggleTranslation}
      style={{
        background: '#4CAF50',
        color: 'white',
        padding: '10px 20px',
        border: 'none',
        borderRadius: '5px',
        margin: '10px',
        cursor: 'pointer'
      }}
    >
      {isUrdu ? 'English میں پڑھیں' : 'اردو میں پڑھیں'}
    </button>
  );
};

export default TranslateButton;