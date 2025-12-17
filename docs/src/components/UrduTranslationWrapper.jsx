import React, { useState } from 'react';
import './UrduTranslation.css';

const UrduTranslationWrapper = ({ children, chapterId }) => {
  const [translationMode, setTranslationMode] = useState('english');

  const urduTranslations = {
    'chapter1': {
      title: 'فزیکل اے آئی کا تعارف',
      content: 'فزیکل اے آئی سے مراد مصنوعی ذہانت کا وہ نظام ہے جو جسمانی دنیا میں کام کرتا ہے...'
    },
    'chapter2': {
      title: 'آر او ایس 2 کی بنیادی باتیں',
      content: 'آر او ایس 2 روبوٹس کے سافٹ ویئر کے لیے ایک مڈل ویئر ہے...'
    },
    'chapter3': {
      title: 'روبوٹ کی ادراکی سسٹم',
      content: 'روبوٹ کی ادراک کے نظام میں کیمرہ، لیزر، اور دیگر حس اشارے شامل ہیں...'
    },
    'chapter4': {
      title: 'گزیبو سیمولیشن',
      content: 'گزیبو ایک طاقتور سیمولیشن انجن ہے جو روبوٹکس کی ترقی کے لیے استعمال ہوتا ہے...'
    },
    'chapter5': {
      title: 'یونٹی کے لیے ایچ آر آئی',
      content: 'یونٹی اور روبوٹکس کے درمیان تعامل کو بہتر بنانے کے لیے یونٹی کا استعمال کیا جاتا ہے...'
    },
    'chapter6': {
      title: 'سینسر کی سیمولیشن',
      content: 'کیمرہ، لیڈار، اور آئی ایم یو کی سیمولیشن کے لیے متعدد ٹولز دستیاب ہیں...'
    },
    'chapter7': {
      title: 'این وی ڈی آئیا سعس ایم',
      content: 'این وی ڈی آئیا سعس ایم ایک پیشرفہ سیمولیشن پلیٹ فارم ہے...'
    },
    'chapter8': {
      title: 'وی ایس ایل اے ایم اور نیوی گیشن',
      content: 'بصری سیمولیٹن اور نقشہ سازی کے لیے وی ایس ایل اے ایم کا استعمال کیا جاتا ہے...'
    },
    'chapter9': {
      title: 'ہیومنوائڈ روبوٹ کی کنیمیٹکس',
      content: 'ہیومنوائڈ روبوٹ کی کنیمیٹکس میں فارورڈ اور انورس کنیمیٹکس شامل ہیں...'
    },
    'chapter10': {
      title: 'بائی پیڈل لوکوموشن کنٹرول',
      content: 'دو پائوں والی چال کے لیے متعدد الگورتھم اور کنٹرول کے طریقے استعمال کیے جاتے ہیں...'
    },
    'chapter11': {
      title: 'مکالماتی اے آئی کا انضمام',
      content: 'صوتی تسلی، قدرتی زبان کی پروسیسنگ، اور چیٹ جی پی ٹی کا انضمام...'
    },
    'chapter12': {
      title: 'ایل ایل ایم - آر او ایس 2 برج کا نفاذ',
      content: 'بڑی زبانی ماڈلز اور روبوٹکس کے درمیان رابطہ قائم کرنے کے لیے برج کا استعمال...'
    },
    'chapter13': {
      title: 'ملٹی ماڈل تعامل کے نظام',
      content: 'بصری، لسانی، اور حرکتی تعامل کے نظام کو مربوط کرنا...'
    },
    'chapter14': {
      title: 'فزیکل اے آئی میں حفاظت اور اخلاق',
      content: 'روبوٹکس میں حفاظتی معیارات اور اخلاقی اصولوں کا تعین...'
    },
    'chapter15': {
      title: ' حقیقی دنیا میں نفاذ',
      content: 'سیمولیشن سے حقیقی دنیا کے اطلاق کے عمل کو سیم ٹو ریل کہا جاتا ہے...'
    },
    'chapter16': {
      title: 'کیپسٹون پروجیکٹ اور مستقبل کے رجحانات',
      content: 'مکمل ہیومنوائڈ روبوٹ پروجیکٹ کا جائزہ اور مستقبل کی ترقیات...'
    }
  };

  const toggleTranslation = () => {
    if (translationMode === 'english') {
      setTranslationMode('urdu');
      document.documentElement.lang = 'ur';
      document.body.classList.add('urdu-mode');
    } else {
      setTranslationMode('english');
      document.documentElement.lang = 'en';
      document.body.classList.remove('urdu-mode');
    }
  };

  return (
    <div className="translation-wrapper">
      <div className="translation-controls">
        <button
          className={`lang-btn ${translationMode === 'english' ? 'active' : ''}`}
          onClick={() => setTranslationMode('english')}
        >
          English
        </button>
        <button
          className={`lang-btn ${translationMode === 'urdu' ? 'active' : ''}`}
          onClick={() => setTranslationMode('urdu')}
        >
          اردو
        </button>
      </div>

      <div className="content-area">
        {translationMode === 'english' ? (
          children
        ) : (
          <div className="urdu-content">
            <h1>{urduTranslations[chapterId]?.title || 'اردو ترجمہ'}</h1>
            <p>{urduTranslations[chapterId]?.content || 'اردو مواد یہاں ہوگا...'}</p>
            <div className="translation-note">
              <small>یہ ڈیمو ہے۔ اصلی ورژن میں مکمل ترجمہ ہوگا۔</small>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UrduTranslationWrapper;