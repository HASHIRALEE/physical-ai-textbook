import React, { type ReactNode } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import RobotReader from '@site/src/components/RobotReader';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const [hoveredChapter, setHoveredChapter] = React.useState('');

  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.robotSection}>
            <RobotReader currentChapter={hoveredChapter} />
          </div>
          <div className={styles.textContent}>
            <Heading as="h1" className={clsx('hero__title', styles.heroTitle)}>
              <span className={styles.gradientText}>{siteConfig.title}</span>
            </Heading>
            <p className={clsx('hero__subtitle', styles.heroSubtitle)}>
              Where Machines Learn to Learn
            </p>
            <div className={styles.heroButtons}>
              <Link
                className="button button--primary button--lg"
                to="/docs/intro">
                🤖 Start Learning with AI
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/docs">
                View All Chapters
              </Link>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

// Computer Chip Style Chapter Cards Component
function ChipCard({ id, number, title, description, link, onHover }: { id: string; number: string; title: string; description: string; link: string; onHover: (id: string) => void }) {
  return (
    <div
      className={styles.chipCard}
      onMouseEnter={() => onHover(id)}
      onMouseLeave={() => onHover('')}
    >
      <div className={styles.chipHeader}>
        <div className={styles.chipNumber}>#{number}</div>
        <div className={styles.chipRobotIcon}>🤖</div>
      </div>
      <Heading as="h3" className={styles.chipTitle}>{title}</Heading>
      <p className={styles.chipDescription}>{description}</p>
      <Link className="button button--secondary button--sm" to={link}>
        Explore Module
      </Link>
    </div>
  );
}

function ChaptersPreview({onChapterHover}: {onChapterHover: (id: string) => void}) {
  const chapters = [
    {
      id: "ch1",
      number: "01",
      title: "Introduction to Physical AI",
      description: "Understanding the fundamentals of Physical AI and embodied intelligence",
      link: "/docs/chapters/ch01-introduction-to-physical-ai"
    },
    {
      id: "ch2",
      number: "02",
      title: "Locomotion Systems",
      description: "Exploring different locomotion mechanisms for humanoid robots",
      link: "/docs/chapters/ch02-locomotion-systems"
    },
    {
      id: "ch3",
      number: "03",
      title: "Perception Systems",
      description: "Sensors, computer vision, and environmental understanding",
      link: "/docs/chapters/ch03-perception-systems"
    },
    {
      id: "ch4",
      number: "04",
      title: "Gazebo Simulation Basics",
      description: "Creating and simulating robotic environments in Gazebo",
      link: "/docs/chapters/gazebo-simulation-basics"
    },
    {
      id: "ch5",
      number: "05",
      title: "Unity for Human-Robot Interaction",
      description: "Building interactive experiences with Unity 3D",
      link: "/docs/chapters/unity-for-human-robot-interaction"
    },
    {
      id: "ch6",
      number: "06",
      title: "Robot Manipulation and Control",
      description: "Arm control, grasping, and manipulation techniques",
      link: "/docs/chapters/robot-manipulation-and-control"
    }
  ];

  return (
    <section className={styles.chaptersSection}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>Learning Modules</Heading>
          <p className={styles.sectionSubtitle}>Interactive modules designed for deep learning</p>
        </div>
        <div className={styles.chipsGrid}>
          {chapters.map((chapter) => (
            <ChipCard
              key={chapter.id}
              id={chapter.id}
              number={chapter.number}
              title={chapter.title}
              description={chapter.description}
              link={chapter.link}
              onHover={onChapterHover}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

// Interactive Features Section
function InteractiveFeatures() {
  return (
    <section className={styles.featuresSection}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>AI-Powered Learning</Heading>
          <p className={styles.sectionSubtitle}>Experience next-generation education with our AI companion</p>
        </div>
        <div className={styles.featuresGrid}>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>🤖</div>
            <Heading as="h3" className={styles.featureTitle}>Ask the Robot</Heading>
            <p className={styles.featureDescription}>Get instant answers to your robotics questions from our AI assistant</p>
          </div>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>📊</div>
            <Heading as="h3" className={styles.featureTitle}>Progress Tracking</Heading>
            <p className={styles.featureDescription}>Track your learning journey with detailed progress analytics</p>
          </div>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>💡</div>
            <Heading as="h3" className={styles.featureTitle}>Smart Tips</Heading>
            <p className={styles.featureDescription}>Receive personalized learning tips based on your progress</p>
          </div>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>🎮</div>
            <Heading as="h3" className={styles.featureTitle}>Interactive Simulations</Heading>
            <p className={styles.featureDescription}>Practice with hands-on simulations in virtual environments</p>
          </div>
        </div>
      </div>
    </section>
  );
}

// Stats Section
function StatsSection() {
  const stats = [
    { number: "16", label: "Comprehensive Modules" },
    { number: "50+", label: "Code Examples" },
    { number: "10+", label: "Simulation Projects" },
    { number: "Beginner to Advanced", label: "Skill Levels" }
  ];

  return (
    <section className={styles.statsSection}>
      <div className="container">
        <div className={styles.statsGrid}>
          {stats.map((stat, index) => (
            <div key={index} className={styles.statItem}>
              <div className={styles.statNumber}>{stat.number}</div>
              <div className={styles.statLabel}>{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Final CTA Section
function CTASection() {
  return (
    <section className={styles.ctaSection}>
      <div className="container">
        <div className={styles.ctaContent}>
          <Heading as="h2" className={styles.ctaTitle}>Ready to Master Robotics?</Heading>
          <p className={styles.ctaSubtitle}>Start your journey to becoming a robotics expert today</p>
          <Link className="button button--primary button--xl" to="/docs/intro">
            Begin Your AI Learning Journey
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  const [hoveredChapter, setHoveredChapter] = React.useState('');

  return (
    <Layout
      title={`Home - ${siteConfig.title}`}
      description="Immersive AI-powered textbook on Physical AI and Humanoid Robotics">
      <HomepageHeader />
      <main>
        <InteractiveFeatures />
        <StatsSection />
        <ChaptersPreview onChapterHover={setHoveredChapter} />
        <CTASection />
      </main>
    </Layout>
  );
}
