import React, { useState, useEffect, useRef } from 'react';
import clsx from 'clsx';
import styles from './RobotReader.module.css';

type RobotReaderProps = {
  isReading?: boolean;
  currentChapter?: string;
  onWave?: () => void;
};

const RobotReader: React.FC<RobotReaderProps> = ({
  isReading = true,
  currentChapter,
  onWave
}) => {
  const [isBlinking, setIsBlinking] = useState(false);
  const [isPointing, setIsPointing] = useState(false);
  const [isExcited, setIsExcited] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [cursorPosition, setCursorPosition] = useState({ x: 200, y: 200 });
  const [isLookingAtCursor, setIsLookingAtCursor] = useState(false);
  const robotRef = useRef<HTMLDivElement>(null);

  // Handle blinking animation
  useEffect(() => {
    const blinkInterval = setInterval(() => {
      setIsBlinking(true);
      setTimeout(() => setIsBlinking(false), 200);
    }, 5000 + Math.random() * 3000); // Random blink interval

    return () => clearInterval(blinkInterval);
  }, []);

  // Handle excitement animation when chapter changes
  useEffect(() => {
    if (currentChapter) {
      setIsExcited(true);
      setTimeout(() => setIsExcited(false), 1000);
    }
  }, [currentChapter]);

  // Handle page turning animation
  useEffect(() => {
    if (isReading) {
      const pageTurnInterval = setInterval(() => {
        setIsPointing(true);
        setTimeout(() => setIsPointing(false), 800);
      }, 10000);

      return () => clearInterval(pageTurnInterval);
    }
  }, [isReading]);

  // Handle cursor following
  const handleMouseMove = (e: React.MouseEvent) => {
    if (robotRef.current) {
      const rect = robotRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setCursorPosition({ x, y });
    }
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
    setIsLookingAtCursor(true);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    setIsLookingAtCursor(false);
  };

  const handleRobotClick = () => {
    if (onWave) onWave();
  };

  // Calculate eye positions based on cursor position for looking effect
  const getEyePosition = (baseX: number, isLeftEye: boolean) => {
    if (!isLookingAtCursor) return baseX;

    const distance = 2; // Max distance eyes can move
    const centerX = 200;
    const centerY = 120;

    const deltaX = cursorPosition.x - centerX;
    const deltaY = cursorPosition.y - centerY;
    const distanceToCursor = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

    if (distanceToCursor < 50) { // Only look if cursor is close enough
      const directionX = deltaX / distanceToCursor;
      return baseX + directionX * distance * 0.5;
    }
    return baseX;
  };

  const leftEyeX = getEyePosition(185, true);
  const rightEyeX = getEyePosition(215, false);

  return (
    <div
      ref={robotRef}
      className={clsx(styles.robotContainer, {
        [styles.robotHovered]: isHovered,
        [styles.robotExcited]: isExcited,
      })}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleRobotClick}
    >
      <svg
        className={styles.robotSvg}
        viewBox="0 0 400 400"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Robot Body */}
        <g className={styles.robotBody}>
          {/* Robot Head */}
          <ellipse
            cx="200"
            cy="120"
            rx="45"
            ry="50"
            fill="#c0c0c0"
            stroke="#a0a0a0"
            strokeWidth="2"
            className={clsx(styles.robotPart, styles.metallic)}
          />

          {/* Robot Eyes - now with dynamic positioning */}
          <circle
            cx={leftEyeX}
            cy={isBlinking ? 120 : 110} // Move down when blinking
            r="8"
            fill={isBlinking ? "#1e293b" : "#60a5fa"}
            className={clsx(styles.eye, {[styles.eyeBlink]: isBlinking})}
          />
          <circle
            cx={rightEyeX}
            cy={isBlinking ? 120 : 110} // Move down when blinking
            r="8"
            fill={isBlinking ? "#1e293b" : "#60a5fa"}
            className={clsx(styles.eye, {[styles.eyeBlink]: isBlinking})}
          />

          {/* LED Lights on Head */}
          <circle
            cx="200"
            cy="85"
            r="3"
            fill="#60a5fa"
            className={clsx(styles.led, {[styles.ledExcited]: isExcited})}
          />

          {/* Robot Body */}
          <rect
            x="160"
            y="170"
            width="80"
            height="100"
            rx="10"
            fill="#c0c0c0"
            stroke="#a0a0a0"
            strokeWidth="2"
            className={clsx(styles.robotPart, styles.metallic)}
          />

          {/* Chest Panel */}
          <rect
            x="170"
            y="180"
            width="60"
            height="40"
            rx="5"
            fill="#1e293b"
            className={styles.panel}
          />
          <rect
            x="175"
            y="185"
            width="50"
            height="30"
            rx="3"
            fill="#0f172a"
            className={styles.screen}
          />

          {/* Button on chest */}
          <circle
            cx="200"
            cy="210"
            r="5"
            fill="#c084fc"
            className={clsx(styles.button, {[styles.buttonExcited]: isExcited})}
          />

          {/* Robot Arms */}
          <rect
            x="140"
            y="180"
            width="20"
            height="60"
            rx="10"
            fill="#c0c0c0"
            stroke="#a0a0a0"
            strokeWidth="2"
            className={clsx(styles.robotPart, styles.metallic, styles.arm, {
              [styles.armPointing]: isPointing,
              [styles.armHovered]: isHovered
            })}
          />
          <rect
            x="240"
            y="180"
            width="20"
            height="60"
            rx="10"
            fill="#c0c0c0"
            stroke="#a0a0a0"
            strokeWidth="2"
            className={clsx(styles.robotPart, styles.metallic, styles.arm)}
          />

          {/* Robot Legs */}
          <rect
            x="170"
            y="270"
            width="25"
            height="50"
            rx="5"
            fill="#c0c0c0"
            stroke="#a0a0a0"
            strokeWidth="2"
            className={clsx(styles.robotPart, styles.metallic)}
          />
          <rect
            x="205"
            y="270"
            width="25"
            height="50"
            rx="5"
            fill="#c0c0c0"
            stroke="#a0a0a0"
            strokeWidth="2"
            className={clsx(styles.robotPart, styles.metallic)}
          />
        </g>

        {/* Book */}
        <g className={clsx(styles.book, {[styles.bookPointing]: isPointing})}>
          {/* Book Cover */}
          <rect
            x="260"
            y="190"
            width="100"
            height="120"
            rx="5"
            fill="#1e293b"
            stroke="#0f172a"
            strokeWidth="2"
            className={styles.bookCover}
          />

          {/* Book Title */}
          <text
            x="310"
            y="240"
            textAnchor="middle"
            fill="#c084fc"
            fontSize="12"
            fontWeight="bold"
            className={styles.bookTitle}
          >
            PHYSICAL
          </text>
          <text
            x="310"
            y="255"
            textAnchor="middle"
            fill="#c084fc"
            fontSize="12"
            fontWeight="bold"
            className={styles.bookTitle}
          >
            AI
          </text>

          {/* Book Pages */}
          <rect
            x="265"
            y="195"
            width="90"
            height="110"
            rx="3"
            fill="#f8fafc"
            stroke="#cbd5e1"
            strokeWidth="1"
            className={styles.bookPages}
          />

          {/* Page Content */}
          <rect
            x="270"
            y="205"
            width="30"
            height="5"
            fill="#cbd5e1"
            className={styles.pageLine}
          />
          <rect
            x="270"
            y="215"
            width="25"
            height="5"
            fill="#cbd5e1"
            className={styles.pageLine}
          />
          <rect
            x="270"
            y="225"
            width="35"
            height="5"
            fill="#cbd5e1"
            className={styles.pageLine}
          />
        </g>

        {/* Floating Tech Icons */}
        <g className={styles.techIcons}>
          {/* Python Icon */}
          <g className={clsx(styles.techIcon, styles.pythonIcon)}>
            <circle cx="80" cy="80" r="15" fill="#3776ab" />
            <path d="M75,70 Q85,65 90,75 Q85,85 75,80" fill="white" />
          </g>

          {/* Neural Network */}
          <g className={clsx(styles.techIcon, styles.nnIcon)}>
            <circle cx="320" cy="60" r="8" fill="#60a5fa" />
            <circle cx="340" cy="60" r="8" fill="#60a5fa" />
            <circle cx="330" cy="45" r="8" fill="#60a5fa" />
            <line x1="320" y1="60" x2="330" y2="45" stroke="#60a5fa" strokeWidth="1" />
            <line x1="340" y1="60" x2="330" y2="45" stroke="#60a5fa" strokeWidth="1" />
          </g>

          {/* Sensor */}
          <g className={clsx(styles.techIcon, styles.sensorIcon)}>
            <circle cx="60" cy="280" r="10" fill="#c084fc" />
            <circle cx="60" cy="280" r="6" fill="#1e293b" />
          </g>

          {/* Code Bracket */}
          <g className={clsx(styles.techIcon, styles.codeIcon)}>
            <path d="M350,250 L345,245 L345,255 Z" fill="#60a5fa" />
            <path d="M360,250 L365,245 L365,255 Z" fill="#60a5fa" />
          </g>
        </g>
      </svg>
    </div>
  );
};

export default RobotReader;