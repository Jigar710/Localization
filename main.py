from collectData import collectData

# userinput
print("0: Collect Data")
print("1: Train")
print("2: Test")
print("3: Dataset")
print("Enter The Option: ")
option = int(input())
if(option == 0):
    collectData()
