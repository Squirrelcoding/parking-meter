import ast
import csv

# Load data from a text file
with open("data/dimensions.txt", "r") as file:
    data = file.read()

# Convert the string to an actual Python list of tuples
points = ast.literal_eval(data)

# Prepare the header and rows
header = ['length', 'width']
rows = [(point[0], point[1]) for point in points]

# Write to a CSV file
with open("output.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)

print("Data has been saved to output.csv")
