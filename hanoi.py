def hanoi(n, source, target, middle):
    global num
    if n == 1:
        print(f"Move 1 from {source} to {target}")
        return
    hanoi(n - 1, source, middle, target)
    print(f"Move {n} from {source} to {target}")
    num += 1
    hanoi(n - 1, middle, target, source)


# 示例调用
num_disks = 3
num = 0
hanoi(num_disks, 'A', 'C', 'B')
print(f"Use {num} steps to attach the target!")