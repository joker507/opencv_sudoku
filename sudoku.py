'''
coding: utf-8
9 * 9 数独求解
'''
m = [[3,0,0,6,0,0,0,8,0],
     [9,8,7,0,1,4,0,0,0],
     [2,0,0,0,3,0,1,0,0],
     [0,7,0,0,0,6,5,0,0],
     [5,0,0,4,0,9,0,0,6],
     [0,0,9,3,0,0,0,2,0],
     [0,0,8,0,4,0,0,0,5],
     [0,0,0,7,6,0,9,1,8],
     [0,5,0,0,0,3,0,0,2]]

#构建数独题目
# m = [[6,0,0,1,0,0,7,0,8],
#      [0,0,0,8,0,0,2,0,0],
#      [2,3,8,0,5,0,1,0,0],
#      [0,0,0,0,4,0,0,9,2],
#      [0,0,4,3,0,8,6,0,0],
#      [3,7,0,0,1,0,0,0,0],
#      [0,0,3,0,7,0,5,2,6],
#      [0,0,2,0,0,4,0,0,0],
#      [9,0,7,0,0,6,0,0,4]]

def Print(m):
    '''打印二维数组'''
    for i in range(len(m)):#行
        for j in range(len(m[i])):#列
            print(m[i][j],end='\t')
        print()

#获取空白格
def get_zero(m):
    '''功能：获取第一个空白'''
    for x in range(9):
        for y in range(9):
            if m[x][y] == 0:
                return x, y
    return -1, -1

#获取下一个空白格子
def get_next_zero(m,x,y):
    '''获取下一个空白,xy为第一个空白格'''
    for next_y in range(y+1, 9):  #同一行
        if m[x][next_y] == 0:
            return x, next_y
    for next_x in range(x+1, 9):  #下一行的全部寻找
        for next_y in range(0, 9):
            if m[next_x][next_y] == 0:
                return next_x, next_y
    return -1, -1

#获取当前要填的数集合
def value(m, x, y):
    '''当前空格要填的集合'''
    #九宫格数字集合
    i = x // 3
    j = y // 3
    grid = [m[i * 3 + k][j * 3 + q] for k in range(3) for q in range(3)]
    v = set([x for x in range(1,10)]) - set(grid) - set(m[x]) - set(list(zip(*m))[y]) #1-9去掉重复的数字(set()：集合无重复性);list(zip(*m))数组转置
    return list(v)

#递归填写空白格
def Try_Sudoku(m, x, y):
    '''功能： 填写空白格
        条件：如果是存在空白格子但是value没有可以填的数，则说明前面的数填错的，则先将此清零返回上一层的Try_sudoku()中尝试下一个v;知道递归符合
    '''
    for v in value(m, x, y):  #递归求解
        m[x][y] = v
        next_x , next_y = get_next_zero(m, x, y)
        if next_y == -1:  #没有下一个空白
            return True
        else:
            juge = Try_Sudoku(m, next_x, next_y) #没有v是返回False
            if juge:    #有下一个空白并且能填
                return True
            m[x][y] = 0 #有空白，没有数字可以填，换下一个v填入，（要将此置零目的是，防止为最后一个v时，并且不符合，此时放回上一层前，应当要将此置零）
    return False#当没有v或者v全都不符合的时候

def solve(n):
    '''数独求解'''
    x, y = get_zero(n)
    Try_Sudoku(n, x, y)
    return n

if __name__ == '__main__':
    solve(m)
    print()
    Print(m)




