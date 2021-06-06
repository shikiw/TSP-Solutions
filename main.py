import os
import time
import argparse
from data_gen import DataGen
from solver import Solver

def main(config):

    # Load data file.
    with open(config.data_dir, "r") as f:
        data = f.read()

    # Solver for solving the TSP problem.
    solver = Solver(input_data=data)

    if config.mode == 'fp':
        t0 = time.clock()
        output = solver.solve_it_by_full_permutation()
        print('time cost by fp: %.6f'%(time.clock() - t0))
        print(output)

    elif config.mode == 'greedy':
        t0 = time.clock()
        output = solver.solve_it_by_greedy()
        print('time cost by greedy: %.6f'%(time.clock() - t0))
        print(output)

    elif config.mode == 'opt2':
        t0 = time.clock()
        output = solver.solve_it_by_opt2()
        print('time cost by opt2: %.6f'%(time.clock() - t0))
        print(output)

    elif config.mode == 'dp':
        t0 = time.clock()
        output = solver.solve_it_by_dp()
        print('time cost by dp: %.6f'%(time.clock() - t0))
        print(output)

    elif config.mode == 'insert':
        t0 = time.clock()
        output = solver.solve_it_by_insert()
        print('time cost by insert: %.6f'%(time.clock() - t0))
        print(output)

    elif config.mode == 'all':
        t0 = time.clock()
        output0 = solver.solve_it_by_full_permutation()
        t1 = time.clock()
        output1 = solver.solve_it_by_greedy()
        t2 = time.clock()
        output2 = solver.solve_it_by_opt2()
        t3 = time.clock()
        output3 = solver.solve_it_by_dp()
        t4 = time.clock()
        output4 = solver.solve_it_by_insert()
        t5 = time.clock()

        print('time cost by fp: %.6f'%(t1 - t0))
        print('time cost by greedy: %.6f'%(t2 - t1))
        print('time cost by opt2: %.6f'%(t3 - t2))
        print('time cost by dp: %.6f'%(t4 - t3))
        print('time cost by insert: %.6f'%(t5 - t4))

        print('\n---------------------fp:')
        print(output0)
        print('\n---------------------greedy:')
        print(output1)
        print('\n---------------------opt2:')
        print(output2)
        print('\n---------------------dp:')
        print(output3)
        print('\n---------------------insert:')
        print(output4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Solution configuration.
    parser.add_argument('--data_type', type=str, default='ours', choices=['ours', 'random'])
    parser.add_argument('--data_dir', type=str, default='city.txt')
    parser.add_argument('--mode', type=str, default='insert', choices=['fp', 'greedy', 'opt2', 'dp', 'insert', 'all'])
    parser.add_argument('--city_num', type=int, default=10, help='the number of cities')
    parser.add_argument('--city_max_distance', type=float, default=10., help='the range of city coordinates')

    config = parser.parse_args()
    print(config)

    # Generate new city data randomly if needed.
    if config.data_type == 'random':
        data_gen = DataGen(config=config)
        data_gen.generate()

    # Solving TSP.
    main(config)