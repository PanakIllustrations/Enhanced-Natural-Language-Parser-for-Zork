# Enhanced Natural Language Parser for Zork

A natural language processing interface for the classic text adventure game Zork, allowing players to use more natural language commands instead of traditional two-word parser syntax.

## Overview

This project enhances the Zork gaming experience by translating natural language input into commands the original Zork parser can understand. Instead of typing `take lamp`, you can say `pick up the lamp` or `I want to grab that lamp`.

## Features

- **Natural Language Processing**: Uses sentence transformers and embedance-based similarity matching
- **Object Affordance Detection**: Automatically infers what actions can be performed with objects
- **Flexible Command Parsing**: Handles various phrasings for the same action
- **Direction Resolution**: Understands movement commands in multiple formats
- **Confidence Scoring**: Provides feedback on parsing certainty

## Dependencies

Install the required Python packages:

```bash
pip install sentence-transformers scikit-learn numpy pexpect
```

## Setup

1. Clone and compile Zork from https://github.com/devshane/zork.git
2. Place both `zork_github_nlp.py` and `zork_nlp_system_9.py` in the same directory as the compiled `zork` executable
3. Ensure the Zork executable is named `zork` (Linux/Mac) or `zork.exe` (Windows)

## Running the Game

```bash
python zork_github_nlp.py
```

The program will automatically locate the Zork executable and start the game with the NLP interface enabled.

## Example Commands

Instead of Zork's traditional syntax, you can use natural language:

- **Movement**: `go south`, `walk north`, `head east`, or just `s`
- **Actions**: `pick up the lamp`, `look at the mailbox`, `open the door`
- **Inventory**: `what am I carrying?`, `check my inventory`
- **Complex actions**: `unlock the door with the key`, `throw the sword at the troll`

## File Structure

- **`zork_github_nlp.py`**: Main game interface that manages the Zork process and handles input/output
- **`zork_nlp_system_9.py`**: Core NLP parser implementing the enhanced natural language processing system
- **`PanakJ Final Report.pdf`**: A written report that summarizes the entire project.
- **`Presentation.pdf`**: Slides for the project video presentation
  
## Technical Details

The NLP system uses:
- Sentence transformers for semantic understanding
- Affordance-based object modeling (based on Fulda et al. 2017)
- Action-object compatibility checking
- Confidence scoring for parse quality assessment

## Controls

- Type commands in natural language
- Use `quit`, `q`, or `exit` to quit the game
- Press Ctrl+C to force quit

## Notes

- The parser will show you the translated Zork command when it differs from your input
- The system learns object affordances dynamically during gameplay
- Confidence scores help indicate parsing accuracy
