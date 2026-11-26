set -e
cd "$(dirname "$0")"
python3 run_sim.py 1000 sim_v3sem_1000.json > sim_v3sem_1000.txt 2>&1
python3 r4check.py 150 > r4check_v3sem_150.txt 2>&1
cp r4check.json r4check_v3sem.json
python3 frontier.py 300 > frontier_v3sem_300.txt 2>&1
cp frontier.json frontier_v3sem.json
echo ALLDONE >> sim_v3sem_1000.txt
