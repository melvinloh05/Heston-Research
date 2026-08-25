width = int(input("input width"))
for i in range(width+1):
    for a in range(i):
        print("*", end = "")
    print()
for i in range(width):
    for j in range(width - i):
        print("*", end = "")
    print()
