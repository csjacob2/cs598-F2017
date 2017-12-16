# Simulator  
Simulator for enterprise documentation research project  


## Time Based Simulator
Select a seed selection heuristic.
```bash
python simulator_time_based.py -m {none,nntu,dtrh,rciw,mmciw,eia}
```

======
## Role Based Simulator  
For different network sizes, what is the minimum number of high degree nodes we have to force to work in order to get 75% of people working at 90% capacity by the 15th iteration?

`python simulator_role_based.py -m nntu`  
or  
`python simulator_role_based.py -m none`


=======  
## Task Based Simulator  
Ran with (Python 2.7.13)  
`python simulator_task_based.py --workers=200 --tasks=200000 --configuration=random --iterations=100`

Or run all heuristics  
`./collect_results__task_based.sh`

task based
