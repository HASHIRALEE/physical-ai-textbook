#!/usr/bin/env python3
"""
Book Viewer Application for Physical AI and Robotics Textbook
Displays all 16 chapters with navigation between them
"""

import os
import re
from pathlib import Path

class BookViewer:
    def __init__(self):
        # Set the chapters directory path
        self.chapters_dir = Path("docs/docs/chapters")
        self.chapters = []
        self.current_chapter_index = 0

    def load_chapters(self):
        """Load all chapter files from the chapters directory"""
        if not self.chapters_dir.exists():
            print(f"Error: Chapters directory does not exist: {self.chapters_dir}")
            return False

        # Find all markdown files in the chapters directory
        chapter_files = list(self.chapters_dir.glob("*.md"))

        # Sort files numerically based on the prefix (01-, 02-, etc.)
        chapter_files.sort(key=lambda x: int(re.match(r'(\d+)-', x.name).group(1)) if re.match(r'(\d+)-', x.name) else 0)

        for file_path in chapter_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract title from the first line if it's a markdown header
            title = self.extract_title(content, file_path.name)

            self.chapters.append({
                'filename': file_path.name,
                'title': title,
                'content': content,
                'number': int(re.match(r'(\d+)-', file_path.name).group(1))
            })

        print(f"Loaded {len(self.chapters)} chapters from {self.chapters_dir}")
        return len(self.chapters) > 0

    def extract_title(self, content, filename):
        """Extract title from the markdown content"""
        lines = content.split('\n')
        for line in lines:
            # Look for markdown headers (lines starting with #)
            if line.strip().startswith('# '):
                return line.strip()[2:]  # Remove '# ' prefix
        # If no header found, use the filename as title
        return filename.replace('.md', '').replace('-', ' ').title()

    def display_menu(self):
        """Display the main menu with all chapters"""
        print("\n" + "="*60)
        print("PHYSICAL AI AND ROBOTICS TEXTBOOK")
        print("="*60)
        print("Select a chapter to read:\n")

        for i, chapter in enumerate(self.chapters):
            print(f"{i+1:2d}. Chapter {chapter['number']:2d}: {chapter['title']}")

        print(f"\n{'-'*60}")
        print("Commands:")
        print("  Enter chapter number (1-{len(self.chapters)}) to read that chapter")
        print("  'all' - List all chapters")
        print("  'quit' or 'q' - Exit the application")
        print("-"*60)

    def display_chapter(self, index):
        """Display a specific chapter"""
        if 0 <= index < len(self.chapters):
            chapter = self.chapters[index]
            print(f"\n{'='*80}")
            print(f"CHAPTER {chapter['number']}: {chapter['title']}")
            print(f"{'='*80}")

            # Display first 20 lines of content (to avoid overwhelming output)
            lines = chapter['content'].split('\n')
            for i, line in enumerate(lines[:50]):  # Show first 50 lines
                print(line)
                if i == 49 and len(lines) > 50:
                    print(f"\n... ({len(lines) - 50} more lines)")
                    break

            # Display navigation options
            self.display_navigation(index)
        else:
            print("Invalid chapter index.")

    def display_navigation(self, current_index):
        """Display navigation options for the current chapter"""
        print(f"\n{'-'*60}")
        print("Navigation:")

        if current_index > 0:
            prev_chapter = self.chapters[current_index - 1]
            print(f"  'p' or 'prev' - Previous: Chapter {prev_chapter['number']}: {prev_chapter['title']}")
        else:
            print("  'p' or 'prev' - Previous: (No previous chapter)")

        print(f"  'menu' - Return to main menu")

        if current_index < len(self.chapters) - 1:
            next_chapter = self.chapters[current_index + 1]
            print(f"  'n' or 'next' - Next: Chapter {next_chapter['number']}: {next_chapter['title']}")
        else:
            print("  'n' or 'next' - Next: (No next chapter)")

        print(f"  'full' - Show full chapter content")
        print("-"*60)

    def display_full_chapter(self, index):
        """Display the full content of a chapter"""
        if 0 <= index < len(self.chapters):
            chapter = self.chapters[index]
            print(f"\n{'='*80}")
            print(f"CHAPTER {chapter['number']}: {chapter['title']}")
            print(f"{'='*80}")
            print(chapter['content'])
            print(f"\n{'='*80}")
            print(f"END OF CHAPTER {chapter['number']}")
            print(f"{'='*80}")

            # Display navigation options
            self.display_navigation(index)
        else:
            print("Invalid chapter index.")

    def run(self):
        """Run the book viewer application"""
        print("Loading Physical AI and Robotics Textbook...")

        if not self.load_chapters():
            print("Failed to load chapters. Please check the directory structure.")
            return

        print(f"Successfully loaded {len(self.chapters)} chapters!")

        while True:
            try:
                # If we're currently viewing a chapter, show just the navigation
                if hasattr(self, 'viewing_chapter') and self.viewing_chapter:
                    user_input = input("\nNavigation command: ").strip().lower()
                else:
                    self.display_menu()
                    user_input = input("\nEnter your choice: ").strip().lower()

                if user_input in ['quit', 'q', 'exit']:
                    print("Thank you for reading! Goodbye!")
                    break
                elif user_input == 'all':
                    self.display_menu()
                elif user_input in ['menu', 'm']:
                    self.viewing_chapter = False
                elif user_input in ['p', 'prev', 'previous']:
                    if self.current_chapter_index > 0:
                        self.current_chapter_index -= 1
                        self.display_chapter(self.current_chapter_index)
                        self.viewing_chapter = True
                    else:
                        print("Already at the first chapter.")
                elif user_input in ['n', 'next']:
                    if self.current_chapter_index < len(self.chapters) - 1:
                        self.current_chapter_index += 1
                        self.display_chapter(self.current_chapter_index)
                        self.viewing_chapter = True
                    else:
                        print("Already at the last chapter.")
                elif user_input == 'full':
                    self.display_full_chapter(self.current_chapter_index)
                    self.viewing_chapter = True
                elif user_input.isdigit():
                    chapter_num = int(user_input)
                    if 1 <= chapter_num <= len(self.chapters):
                        self.current_chapter_index = chapter_num - 1
                        self.display_chapter(self.current_chapter_index)
                        self.viewing_chapter = True
                    else:
                        print(f"Please enter a number between 1 and {len(self.chapters)}")
                else:
                    print("Invalid input. Please try again.")

            except KeyboardInterrupt:
                print("\n\nThank you for reading! Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Returning to main menu...")

def main():
    viewer = BookViewer()
    viewer.run()

if __name__ == "__main__":
    main()