# The project

This project is a text classifier using Rust and the Burn library for training machine learning models. 
The project includes data pre-processing, vocabulary creation and model training.

## Main files

- `src/main.rs`: Program entry point. Loads data, preprocesses examples, initializes and trains the model.
- `src/dataset.rs`: Defines data structure and methods for loading data from a JSON file.
- `src/preprocess.rs`: Contains functions for data pre-processing, including tokenization and padding.
- `src/vocabulary.rs` : Manages vocabulary creation and encoding.
- `src/model.rs`: Defines the structure of the text classification model.

## Usage

1. Make sure you have Rust installed on your machine.
2. Clone the repository.
3. Place your data file `dataset.json` in the project root directory.
4. Compile and run the project with the command :


## Example data

{
    “categoryExample": [
        {
            “text": ‘Example text to classify’,
            “label": 0,
            “is_valid": true
        },
        ...
    ]
}

## Contributions

Contributions are welcome! Feel free to open an issue or propose a pull request.