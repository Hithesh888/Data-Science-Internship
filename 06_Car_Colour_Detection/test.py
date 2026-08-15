from detect import detect_objects

output, cars, people = detect_objects(r"D:\intern\project6\dataset\test.jpeg")

print("Cars:", cars)
print("People:", people)
print("Saved:", output)