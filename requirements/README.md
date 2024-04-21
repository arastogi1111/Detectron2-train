# Requirements

## Detectron 2 Training

Install relevant Detectron 2 [version](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only) after checking cuda and torch versions.

To reproduce the enviroment, there are several options provided to try (arranged as clean to bloated):

##### req_pigar.txt

 - Uses [Pigar](https://github.com/damnever/pigar) package to generate only the requirements in accordance with imports in code.
 - Only considers the imports of the specific repo (Detectron2_train in this case) 
 - This boils it down to the minimum installations and reduces possibilities of conflicts.
 - The txt file can be used normally via pip to recreate environment
 - You may require to approve and/or initiate further installations if prompted, depending on your base installations.

##### req_pipreqs.txt

 - Uses [pipreqs](https://github.com/bndr/pipreqs) package to generate requirements from imports.
 - --use-local flag was used.

#####  env_no_builds.yml

 - conda export without build strings
 - suggested [here](https://stackoverflow.com/a/64501566/7442793)
 - Useful for OS-agnostic installation


### More bloated installations : (irrelevant packages could be present)

##### env.yml
 - usual full conda export of environment with builds specific to original platform.
 
##### req_piplist.txt
 - frozen pip list. 
 - suggested [here](https://stackoverflow.com/a/64501566/7442793)

##### req_pipfreeze.txt
 - usual pip freeze of full environment, could fail to resolve directories.
 - only records and saves the packages that were installed with pip install.
 

 
 
 

 
