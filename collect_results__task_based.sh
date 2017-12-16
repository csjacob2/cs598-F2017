#!/usr/bin/env bash
rm -rf results/ && python simulator_task_based.py --workers=200 --tasks=200000 --configuration=random --iterations=100
rm -rf results/ && python simulator_task_based.py --workers=200 --tasks=200000 --configuration=per_team --iterations=100
rm -rf results/ && python simulator_task_based.py --workers=200 --tasks=200000 --configuration=sparse --iterations=100
rm -rf results/ && python simulator_task_based.py --workers=200 --tasks=200000 --configuration=smallest --iterations=100
rm -rf results/ && python simulator_task_based.py --workers=200 --tasks=200000 --configuration=largest --iterations=100
rm -rf results/ && python simulator_task_based.py --workers=200 --tasks=200000 --configuration=weakest --iterations=100
rm -rf results/ && python simulator_task_based.py --workers=200 --tasks=200000 --configuration=strongest --iterations=100
