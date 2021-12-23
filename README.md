
This repo contains code for EMNLP 2021 paper: 
**Uncovering Implicit Gender Bias Through Commonsense Inference**


## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Prerequisites
  ```sh
  pip install -r requirements.txt
  ```
Download RocStory dataset from https://cs.rochester.edu/nlp/rocstories/

Download StanfordNERTagger

### Installation

1. COMeT2

   Install COMeT2 according to https://github.com/vered1986/comet-commonsense

<!-- USAGE EXAMPLES -->
## Usage

1. Classify stories according to protagonsit's gender
      ```sh
      python preprocess.py <story_filename.tsv>
      ```
2. Anonymization
      
      ```sh
      python replaceGender.py 
      ```      
3. Extract stories having more than two characters

      ```sh
      python extractTwo.py 
      ```  
   
4. Classify sentences according to protagonist
      ```sh
      python findSubj.py 
      ```  
   
5. Get COMeT outputs

      ```sh
      python generate_inferences.py
      ```  
   
6. Calculate Valence, arousal scores 
      ```sh
      python connotation_COMET_NRC.py
      ```  
   
7. Calculate Intellect, Appearance, Power scores
      ```sh
      python get_lexicon_average.py
      ```  
      
      Acknowledgement:
      
      We borrowed some code from this repository: https://github.com/ddemszky/textbook-analysis
