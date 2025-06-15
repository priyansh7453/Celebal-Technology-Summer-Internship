def lower_pattern(n):
    for i in range(n):
        for j in range(i+1):
                print("*", end=" ")  # Print '*' or any character you prefer
        print()

n = 5  # Size of the pattern
lower_pattern(n)
