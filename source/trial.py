file = open(r"../data/nana.csv")
result = []
for line in file:
    result.append(line)
print(set(result))
