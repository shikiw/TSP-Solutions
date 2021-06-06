# TSP-Solutions

Some classical solutions for City Travel TSP problem.

## Usage.

To use your own fixed city data and solve it with the desired solution (e.g., dp):

`python main.py --data_dir city.txt --mode dp`

To use your own fixed city data and solve it with all solutions:

`python main.py --data_dir city.txt --mode all`

To generate the new city data randomly and solve it with the desired solution (e.g., dp):

`python main.py --data_type random --city_num 10 --mode dp`

To generate the new city data randomly and solve it with all solutions:

`python main.py --data_type random --city_num 10 --mode all`
