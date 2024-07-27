T = int(input())
nc = []
stm = []
for i in range(T):
    nc.append(map(int, input().strip().split()))
q = int(input())

def get_num(num):
    find_n = []
    while num > 0:
        n = num % 2
        find_n.append(n)
        num = num // 2
    return find_n
js = 0
for i in range(q):
    stm.append(map(int, input().strip().split()))
    for num_1 in range(stm[q][0]-1, stm[q][1]+1):
        mask = get_num(stm[q][2])
        num_2 = get_num(num_1)
        for b in range(len(mask)):
            if mask[b] == 1 and num_2[b] != 1:
                js += 1
                break

    print(stm[q][1]-stm[q][1]+1-js)





