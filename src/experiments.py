import subprocess

runs = [
    'python3 main.py --model_name=test1 --conv_cond=1 --batch_size=300',
    'python3 main.py --model_name=test2 --conv_cond=1 --batch_size=300'
]

for run in runs:
    subprocess.run(run, shell=True)
    for i in range(10):
        print('NEXT MODEL!\n')

print('NO MORE MODELS..... :-(')