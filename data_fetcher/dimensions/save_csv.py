import ast
import csv

with open("data/dimensions.txt", "r") as file:
    data = file.read()

points = ast.literal_eval(data)

header = ['length', 'width']
rows = [(point[0], point[1]) for point in points]

with open("output.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)

print("Data has been saved to output.csv")
