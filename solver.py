import math
import numpy as np
from collections import namedtuple

class Solver(object):
    def __init__(self, input_data=None):
        '''
        the ensemble of different solutions for TSP problem, including full permutation.
        input_data: city data file, the first row indicates the city number, the rest means the coordinates of cities.
        '''
        self.input_data = input_data
        self.lines = self.input_data.split('\n')
        self.nodeCount = int(self.lines[0])
        self.Point = namedtuple("Point", ['x', 'y'])
        self.points = self.inputParse()
        self.dis = self.point2dis(self.points)
    
    def permutation(self, xs):
        '''
        the function of generating the full permutation.
        xs: range list.
        '''
        if len(xs) == 0 or len(xs) == 1:
            return [xs]
        result = []
        for i in xs:
            temp_list = xs[:]
            temp_list.remove(i)
            temp = self.permutation(temp_list)
            for j in temp:
                j[0:0] = [i]
                result.append(j)
        return result
    
    def length(self, point1, point2):
        '''
        calculate the distance between the given two self.points.
        point1: x1, y1.
        point2: x2, y2.
        '''
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def generatePath(self, path):
        l = len(path)
        a = np.random.randint(0, l)
        while True:
            b = np.random.randint(0, l)
            if np.abs(a - b) > 1:
                break
        if a > b:
            return b, a, path[b:a+1]
        else:
            return a, b, path[a:b+1]

    def reversePath(self, path):
        re_path = path.copy()
        re_path[1:-1] = re_path[-2:0:-1] # reverse the path indicated by [1:-1].
        return re_path

    def calPath(self, path, dis):
        s = 0.0
        l = len(path)
        for i in range(0, l-1):
            # print(path[i])
            # print(path[i+1])
            # print(dis)
            s += dis[path[i]][path[i+1]]
        return s

    def comparePath(self, path1, path2, dis):
        if self.calPath(path1, dis) < self.calPath(path2, dis):
            return True
        else:
            return False
    
    def inputParse(self):
        # parse the input
        points = []
        for i in range(1, self.nodeCount+1):
            line = self.lines[i]
            parts = line.split()
            points.append(self.Point(float(parts[0]), float(parts[1])))
        return points

    def point2dis(self, points):
        # distance matrix (upper-triangular matrix, convenient for subsequent processing).
        # complete graph, any two places can be reached.
        n = float("inf")
        dis = [[n for col in range(self.nodeCount)] for row in range(self.nodeCount)]
        for i in range(self.nodeCount-1):
            for j in range(i+1, self.nodeCount):
                dis[i][j] = self.length(points[i],points[j])
        for i in range(1, self.nodeCount):
            for j in range(0, i):
                dis[i][j] = dis[j][i]
        return dis

    def solve_it_by_full_permutation(self):
        '''the solution based on full permutation.'''

        # select the smallest in the full permutation.
        solution_sta = list(range(0, self.nodeCount))
        solutions = self.permutation(solution_sta)
        obj_best = float("inf")
        i = 0
        for s in solutions:
            obj = self.length(self.points[s[-1]], self.points[s[0]])
            i += 1
            for index in range(0, self.nodeCount-1):
                obj += self.length(self.points[s[index]], self.points[s[index+1]])
            # print(s, obj)
            if obj < obj_best:
                obj_best = obj
                solution = s
    
        # prepare the solution in the specified output format.
        output_data = '%.2f' % obj_best + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, solution))

        return output_data

    def solve_it_by_greedy(self):
        '''the solution based on greedy algo.'''
            
        # greedy algorithm
        flag = [0] * self.nodeCount   # noted the node
        flag[0] = 1
        solution = [0] * self.nodeCount
        obj = 0
        i = 0
        index = 0
        while i < self.nodeCount-1:
            dis_min = float("inf")
            for j in range(0, self.nodeCount):
                if flag[j] == 0 and j != index:
                    dis = self.length(self.points[index], self.points[j])
                    if dis < dis_min:
                        dis_min = dis
                        index0 = j
            index = index0
            solution[i+1] = index
            flag[index] = 1
            obj += dis_min
            i += 1
        obj += self.length(self.points[0], self.points[index])
        
        output_data = '%.2f' % obj + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, solution))

        return output_data

    def solve_it_by_opt2(self):
        '''the solution based on opt2.'''

        ##greedy algorithm(hamilton circle).

        # opt-2     
        # build a trivial solution.
        bestPath = list(range(0, self.nodeCount))
        bestPath.append(0)
        
        if self.nodeCount == 51:
            MAXCOUNT = 200
        else:
            MAXCOUNT = 10  
        
        count = 0
        while count < MAXCOUNT:
            start, end, path = self.generatePath(bestPath)  # generate a random path
            rePath = self.reversePath(path)
            if self.comparePath(path, rePath, self.dis):
                count += 1
                continue
            else:
                count = 0
                bestPath[start:end+1] = rePath
        
        obj = 0.0
        obj += self.calPath(bestPath, self.dis)
        bestPath.pop()
        solution = bestPath
        output_data = '%.2f' % obj + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, solution))

        return output_data

    def solve_it_by_dp(self):
        '''the solution based on dp.'''

        # dis is the distance metric.
        n = float("inf")
        cnt = 2 ** (self.nodeCount-1)
        dp = [[n for col in range(cnt)] for row in range(self.nodeCount+1)]
        for i in range(1, self.nodeCount):
            dp[i][0] = self.dis[i][0]

        for i in range(1, cnt-1):          # the set V
            for j in range(1, self.nodeCount):
                if i&(1<<(j-1)) == 0:      # if j is not in the set V
                    for k in range(1, self.nodeCount):
                        if (1<<(k-1))&i:   # if k is in the set V
                            dp[j][i] = min(dp[j][i], self.dis[j][k]+dp[k][i-(1<<(k-1))])

        for k in range(1, self.nodeCount):
            if (1<<(k-1))&(cnt-1):   # if k is in the set
                    dp[0][cnt-1] = min(dp[0][cnt-1], self.dis[0][k]+dp[k][cnt-1-(1<<(k-1))])

        obj = dp[0][cnt-1]  
        # print(obj)

        return obj
    
    def solve_it_by_insert(self):
        '''the solution based on insert.'''

        s = np.random.randint(self.nodeCount)  # random selection of initial nodes
        V = list(range(self.nodeCount)) # all node sets
        Vt = [s]                   # visited node
        Vr = list(set(V)^set(Vt))  # node not accessed
        Et = [(s,s)]               # edge
        w = [[0 for col in range(self.nodeCount)] for row in range(self.nodeCount)]
        n = float("inf")
        for i in range(self.nodeCount):
            for j in range(self.nodeCount):
                if j != i:
                    w[i][j] = self.dis[i][j]

        tweight = 0

        dist = self.dis[s]
        while len(Vt) < self.nodeCount:
            
            # selection
            max_ = -n

            for i in range(self.nodeCount):
                if dist[i] > max_ and dist[i] != n:
                    max_ = dist[i]
                    f = i
            # insert
            min_ = n
            for edge in Et:
                c = w[edge[0]][f] + w[f][edge[1]] - w[edge[0]][edge[1]]
                if c < min_:
                    min_ = c
                    row = edge[0]
                    col = edge[1]
            Et.remove((row, col))
            Et.append((row, f))
            Et.append((f, col))
            Vt.append(f)
            Vr.remove(f)
            tweight += min_
            dist[f] = n 
            # change the distance metric.
            for x in Vr:
                dist[x] = min(dist[x], w[f][x])


        route = [0] * self.nodeCount
        for i in range(1, self.nodeCount):
            for j in range(self.nodeCount):
                if route[i-1] == Et[j][0]:
                    route[i] = Et[j][1]
                    break
        
        obj = self.length(self.points[route[-1]], self.points[route[0]])
        for index in range(0, self.nodeCount-1):
            obj += self.length(self.points[route[index]], self.points[route[index+1]])
        
        output_data = '%.2f' % obj + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, route))

        return output_data

