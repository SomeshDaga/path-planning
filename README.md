# Path Planning

## Usage

If setting up for the first time, follow these instructions to run the demos:

1. Install python dependencies:
    - If using conda (install + activate environment):
        
            conda create env -f environment.yml
            conda activate lab3
     
    - If using pip (requires python 3.6):
    
            pip install -r requirements.txt
        
2. Navigate to the working directory of the demos:

        cd path_planning
        
## Demos

There are demos for multiple path planning algorithms:

### Potential Field

        python pf_demo.py
        
### Probabilistic Roadmap (PRM)

        python prm_demo.py
        
### Rapidly-exploring Random Trees (RRT)

A bidirectional tree search version of RRTs (RRT-Connect) is implemented here

        python rrt_demo.py