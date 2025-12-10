# Data Model: Physical AI and Humanoid Robotics Textbook

## Core Entities

### Textbook Chapter
- **ID**: Unique identifier (e.g., "ch01-introduction-to-physical-ai")
- **Title**: Chapter title (string)
- **Subtitle**: Optional subtitle (string, nullable)
- **WordCount**: Target 2000 words (integer, default: 2000)
- **LearningObjectives**: Array of learning objectives (array of strings)
- **CoreConcepts**: Theoretical foundations content (string, markdown format)
- **HandsOnTutorial**: Step-by-step tutorial content (string, markdown format)
- **CodeImplementation**: Python/ROS2 code examples (string, markdown with code blocks)
- **Exercises**: Array of exercises and challenges (array of objects)
- **FurtherReading**: Array of references and resources (array of strings)
- **ChapterSummary**: Key takeaways (string, markdown format)
- **PartNumber**: Part in the 4-part curriculum (integer, 1-4)
- **WeekNumber**: Week in the 16-week curriculum (integer, 1-16)
- **Prerequisites**: Array of prerequisite concepts (array of strings)
- **RelatedChapters**: Array of related chapter IDs (array of strings)
- **CreatedAt**: Creation timestamp (datetime)
- **UpdatedAt**: Last modification timestamp (datetime)
- **Author**: Author information (string)
- **ReviewStatus**: Draft, Review, Approved, Published (enum)

### Exercise
- **ID**: Unique identifier (string)
- **Title**: Exercise title (string)
- **Description**: Detailed exercise description (string, markdown format)
- **DifficultyLevel**: Beginner, Intermediate, Advanced (enum)
- **Type**: Theoretical, Implementation, Analysis, Design (enum)
- **RequiredTime**: Estimated completion time in minutes (integer)
- **LearningObjectives**: Array of related learning objectives (array of strings)
- **Solution**: Reference solution or guidance (string, markdown format)
- **Hints**: Array of hints for students (array of strings)
- **ChapterID**: Reference to parent chapter (string)
- **CreatedAt**: Creation timestamp (datetime)
- **UpdatedAt**: Last modification timestamp (datetime)

### CodeExample
- **ID**: Unique identifier (string)
- **Title**: Example title (string)
- **Description**: Brief description of the example (string)
- **Language**: Python, C++, ROS2 launch, etc. (enum)
- **Code**: The actual code content (string)
- **Explanation**: Explanation of the code (string, markdown format)
- **Dependencies**: Required packages or libraries (array of strings)
- **SimulationEnvironment**: Gazebo, Isaac Sim, etc. (string, nullable)
- **ChapterID**: Reference to parent chapter (string)
- **CreatedAt**: Creation timestamp (datetime)
- **UpdatedAt**: Last modification timestamp (datetime)

### UserProgress
- **UserID**: Unique user identifier (string)
- **ChapterID**: Chapter being tracked (string)
- **Status**: Not Started, In Progress, Completed (enum)
- **CompletionPercentage**: 0-100 (integer)
- **TimeSpent**: Time spent in minutes (integer)
- **LastAccessed**: Last access timestamp (datetime)
- **ExerciseProgress**: Array of exercise completion status (array of objects)
- **CreatedAt**: Creation timestamp (datetime)
- **UpdatedAt**: Last modification timestamp (datetime)

### ROS2Component
- **ID**: Unique identifier (string)
- **Name**: Component name (string)
- **Type**: Node, Package, Message, Service, Action (enum)
- **Description**: Component description (string)
- **Parameters**: Configuration parameters (object)
- **TopicsPublished**: Array of published topics (array of strings)
- **TopicsSubscribed**: Array of subscribed topics (array of strings)
- **Services**: Array of provided services (array of strings)
- **Actions**: Array of provided actions (array of strings)
- **Dependencies**: Required ROS2 packages (array of strings)
- **ChapterID**: Reference to parent chapter (string)
- **CreatedAt**: Creation timestamp (datetime)
- **UpdatedAt**: Last modification timestamp (datetime)

### SimulationModel
- **ID**: Unique identifier (string)
- **Name**: Model name (string)
- **Type**: Robot, Environment, Object (enum)
- **Description**: Model description (string)
- **URDFPath**: Path to URDF file (string, nullable)
- **SDFPath**: Path to SDF file (string, nullable)
- **ConfigFiles**: Array of configuration file paths (array of strings)
- **Parameters**: Model parameters (object)
- **ChapterID**: Reference to parent chapter (string)
- **CreatedAt**: Creation timestamp (datetime)
- **UpdatedAt**: Last modification timestamp (datetime)

## Relationships

### Chapter -> Exercises
- One chapter contains many exercises (1 to many)

### Chapter -> CodeExamples
- One chapter contains many code examples (1 to many)

### Chapter -> ROS2Components
- One chapter references many ROS2 components (1 to many)

### Chapter -> SimulationModels
- One chapter references many simulation models (1 to many)

### User -> UserProgress
- One user has many progress records (1 to many)

### UserProgress -> Exercises
- One progress record tracks many exercises (1 to many)

## Validation Rules

### Textbook Chapter
- WordCount must be between 1800 and 2200 words
- LearningObjectives must contain at least 3 objectives
- CoreConcepts must not be empty
- ChapterSummary must not be empty
- PartNumber must be between 1 and 4
- WeekNumber must be between 1 and 16

### Exercise
- DifficultyLevel must be one of the allowed values
- RequiredTime must be greater than 0
- Solution or Hints must be provided

### CodeExample
- Language must be one of the supported languages
- Code must be valid syntax for the specified language

### UserProgress
- Status must be one of the allowed values
- CompletionPercentage must be between 0 and 100
- UserID and ChapterID combination must be unique

## State Transitions

### Chapter Review Status
- Draft → Review (when content is ready for review)
- Review → Approved (when reviewed and approved)
- Approved → Published (when ready for release)
- Published → Approved (when updates are made to published content)

### Exercise Progress
- Not Started → In Progress (when student starts exercise)
- In Progress → Completed (when student completes exercise)
- Completed → In Progress (when student revisits exercise)