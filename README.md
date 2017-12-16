#Simulator  
Simulator for enterprise documentation research project  




##Role Based Simulator  
For different network sizes, what is the minimum number of high degree nodes we have to force to work in order to get 75% of people working at 90% capacity by the 15th iteration?

`python simulator_role_based.py -m nntu`  
or  
`python simulator_role_based.py -m none`

// I don't think we can use DTRH. We can't peek at people's thresholds.  
// perhaps record the numbers so we can perform significance tests.  
// did nntu, dtrh, and rciw. Do the other stuff.  


=======  
##Task Based Simulator  
Ran with (Python 2.7.13)  
`python simulator_task_based.py --workers=200 --tasks=200000 --configuration=random --iterations=100`

Or run all heuristics  
`./collect_results__task_based.sh`

task based
