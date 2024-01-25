# Projet Master

Members : Filipe AUGUSTO, Mohamed BENZAHIA, Luc BURCKEL, Iman IRAJDOOST, Caleb MUKAYA GAKALI, Evan REGNAULT, Vincent WENDLING

Professor : Julien GOSSA

# Description

A project to estimate house prices using AI models. This project is part of the Master Project SIL at the University of Strasbourg.

# Installation

## Requirements

- Python 3.9
- pip

## Installation

To install the project, you need to clone the repository and install the requirements.

```bash
pip install -r requirements.txt
```

# Usage

To use the project, you need to run the main.py file.

```bash
python3 main.py <city_name> [-a] [-t <model_name>] [-p <path_to_json_file>]
```

The city name is the name of the city where you want to estimate the price of a house. 


Supported cities are 
``
[
    'bordeaux',
    'lille',
    'lyon',
    'marseille',
    'montpellier',
    'nantes',
    'nice',
    'paris',
    'strasbourg',
    'toulouse'
]
``


-a is an optional argument to use when you want the application to show the analysis of the data.


-t is an optional argument to use when you want to use a specific model. Supported models are

``
[
    'gbr',
    'random_forest
]
``


-p is an optional argument to use when you want to use a specific json file to estimate the price of a house. Follow -p with the path to the json file.



# Tools

## Mean

Calculates the mean of the data without using any model.

### Usage

```bash
python3 mean.py <city_name>
```

## Reduce File

Reduces the size of the data file by only keeping the given number of lines (for testing purposes).

### Usage

```bash
python3 reduce_file.py <element_count> <input_file_path> <output_file_name>
```

# License

[MIT](https://choosealicense.com/licenses/mit/)

MIT License

Copyright (c) [2024] [Iman Irajdoost, Caleb Mukaya, Vincent Wendling, Luc Burckel, Filipe Augusto, Mohamed Benzahia, Evan Regnault]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.