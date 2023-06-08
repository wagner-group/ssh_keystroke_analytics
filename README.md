# Network Detection of Interactive SSH Impostors Using Deep Learning

This repo contains the code for training and evaluating SSH keystroke authentication models. 
See our paper for more details.

## Installation

* Clone the repo
* Setup a virtual environement: `python3 -m venv env && source env/bin/activate`
* Run the installation scripts `./install.sh`


## Usage


`keystroke_analytics [type_of_model] [data_format] [data_directory] [options]` 

* type_of_model: sets the model type. Options are `authentication`, `classification`, `classification_without_others`, `TypeNet`, `CNNGRU` and `FSNet`. Use `authentication` to reproduce the paper model.
* data_format: set the type of data partitioning. Use `keystrokes_alternate` to reproduce the paper
* data_directory: path to the data

There is a long list of options to tune your model:
* --testing-dir str+:i paths to testing datasets
* --detailed-eval: run a detailed evaluation (recommended)
* -N int: Max length of SSH connection (default: 512)
* -n int: Min length of SSH connection
* -B int: Batch size (default: 32)
* -P int: Number of partitions (default: 8)
* -p int: Number of bins per partition (default: 8)
* -e int: Positional encoding dimension (default: -1, meaning it is set automatically)
* -L float: Learning rate (default: 2e-5)
* -l str: Path to model to load (when fine-tuning or evaluating)
* -R int: Randomness seed
* -S int: Use packet sizes (default: 1 (for true))
* -T int: Use packet inter-arrival times (default: 1 (for true))
* -E int: Number of training epochs (default: 15). Set to 0 for running eval only
* --thead int: Number of transformer heads (default: 4)
* --tlayer int: Number of transformer layers (default: 4)
* --predict: Run prediction only
* -G float: Learning rate decay (default: 0.95)
* -H int: Learning rate scheduler step (default: 5)
* -O str: Output folder
* -g int: GPU to use


## Data format

Our data format allows to handle potentially large datasets that cannot fit in GPU memory or CPU memory, at the cost of being structured in an inconvenient format. 
Data must be placed in a separate folder, with one file per user, one file for the other users that do not have enough traffic to be counted individually, and a manifest file.

### User files
Each user file is a TSV, with the following fields:
`USER[int]  TIMESTAMP[float]    SOURCE_IP[vec]  DEST_IP[vec] SEQUENCE_OF_INTERARRIVAL_TIMES[vec of float]   SEQUENCE_OF_LENGTHS[vec of intervals]   POSITION[float]`

* USER: integer identifier for the user
* TIMESTAMP: timestamp of the start of the connection [not used in classification]
* SOURCE_IP: source ip, represented as a vector of bit values. Is not used in classification, so can be set to 0
* DEST_IP: destination ip. Can be set to 0, like source
* SEQUENCE_OF_INTERARRIVAL_TIMES: vector of inter-arrival times. Separate values with commas: eg 0.3,0.01,0.05...
* SEQUENCE_OF_LENGTHS: vector of packet size intervals. Separate value with commas: eg 10:30,20:25,50:60
* POSITION: relative position (as a real value between 0 and 1) of the sequence in the full connection, if this is a truncated connection missing the start

each user file must be named after the user integer identifier, and the others file must be named others. 

### Manifest
Manifest file is a TSV named `authentication_manifest`, with the following fields: 
`FILE[str] COUNT[int]`

* FILE: filename for each user file (either a user identifier, or "others"
* COUNT: number of lines in the file

### Training / Eval / Testing
One folder must be created for training, testing and eval. The data_path is the path to the folder containing training and eval, while the testing path is just the path of the testing folder. 

### Example
Folder `examples/data` contains an example of the data format, with fake data. 
