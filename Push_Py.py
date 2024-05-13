"""
print("Ador Bhuna")
print("New_line")
print("Changed Line")
"""
"""
colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]

for i in colors:
    print(i)
"""
"""
for i in range(1,6):
    print(i)
"""
colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]

for i in colors:
    print(i)
"""
"""

for number in range(50,100):
    print(number)
"""

"""
fruits = ["apple", "banana", "orange"]

for index, fruit in enumerate(fruits):
    print(f"At position {index}, I found {fruit}")
"""
"""
count = 1

while count <= 10:
    print(count)
    count += 2
"""
"""
numbers = [1,2,3,4,5,6,7,8,9,10]
for count in numbers:
    print(count)
"""
"""
dates = [1982,1980,1973]
N = len(dates)

for i in range(N):
    print(dates[i])
"""
"""
colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
N = len(colors)

for i in range(N):
    print(colors[i])
"""
"""
squares = ['Red','Yellow','Black','Blue','Green']

for i in range(0,5):
    print("Before Square ",i,"is", squares[i])
    squares[i] = 'White'
    print("After Square ",i,"is", squares[i])
"""
"""
# While Loop Example

dates = [1982, 1980, 1973, 2000]

i = 0
year = dates[0]

while year != 1973:
    i = i+1
    year = dates[i]
    

print("It took ", i ,"repetitions to get out of loop.")

"""
"""
PlayListRatings = [10, 9.5, 10, 8, 7.5, 5, 10, 10]
i = 0
Rating = PlayListRatings[0]
while(i < len(PlayListRatings) and Rating >= 6):
    print(Rating)
    i = i + 1 # This prints the value 10 only once 
    Rating = PlayListRatings[i]
    i = i + 1 #Try uncommenting the line and comment the previous i = i + 1, and see the difference, 10 value will get printed twice because when the loop starts it will print Rating and then with PlayListRatings[0], it will again assign the value 10 to Ratings.
"""
"""
squares = ['orange', 'orange', 'purple', 'blue ', 'orange']
new_squares = []
i = 0
while(i < len(squares) and squares[i] == 'orange'):
    new_squares.append(squares[i])
    i = i + 1
print (new_squares)

"""
"""
def mult(a,b):
 
    c = a*b;
    return c

print(mult(5,"Ador"))
"""
"""
def cat(*names):
    for name in names:
        print(name)

cat("Ador","Bhuna")
"""
"""
#Global Scope

def thriller():
    date = 1982
    return (date)

#Global Scope
date = 2017 #Global Variable

#Calling variable from local scope
print(thriller())

#Calling variable from global scope
print(date)

"""

"""
#Declare Global Variable

def local_cat():
    global cat
    cat = "Ador"

    return cat


print(local_cat())

#calling global variable from local scope

print(cat)

"""
"""
def calculate_total(a,b):
    total = a + b
    return total
    
result = calculate_total(5,7)
print(result)
"""
"""
def greet(name):
    print("Hello, "+ name)

result = greet("Alice")
print(result)  # Output: Hello, Alice
"""
"""
def multiply(a, b):
    """
"""
    This function multiplies two numbers.
    Input: a (number), b (number)
    Output: Product of a and b
    """
"""
    print(a * b)
multiply(2,6)
help(multiply)
"""
"""
def example_function():
    local_variable = "I'm local"
    print(local_variable)   # Accessing local variable
global_variable = "I'm Global"

print(global_variable)  # Accessing global variable
print(example_function())
"""
"""
def example_function():
    local_variable = "I'm local"
    print(global_variable)  # Accessing global variable
    print(local_variable)   # Accessing local variable
global_variable = "I'm Global"
print(example_function())
print(global_variable)
"""
"""
def print_numbers(limit):
    for i in range(1, limit+7):
        print(i)
print_numbers(5)
"""
"""
def greet(name):
    return "Hello, " + name

for i in range(3):
    print(greet("Alice"))
"""
"""
# Define an empty list as the initial data structure
my_list = []
# Function to add an element to the list
def add_element(data_structure, element):
    data_structure.append(element)

# Add elements to the list using the add_element function
add_element(my_list, 42)
add_element(my_list, 17)
add_element(my_list, 99)

print(my_list)
"""
"""
my_list = []

def add_element(data_structure, element):
    data_structure.append(element)

add_element(my_list, 24)
add_element(my_list, 23)
add_element(my_list, 26)
print(my_list)
"""
"""
my_list = [24,25,27]
# Function to remove an element from the list
def remove_element(data_structure, element):
    if element in data_structure:
        data_structure.remove(element)
    else:
        print(f"{element} not found in the list.")
remove_element(my_list,78)
print (my_list)
"""
"""
In the given code snippet, `f"{element} not found in the list."` is an example of an f-string. 

An f-string is a feature in Python that allows you to embed expressions inside string literals, using curly braces `{}`. When the string is prefixed with `f` or `F`, Python replaces expressions inside curly braces with their values. 

In the context of the code, `{element}` inside the f-string gets replaced with the value of the `element` variable. So, if `element` is 78, the string `"78 not found in the list."` will be printed. 

This is a convenient way to compose strings dynamically, incorporating variables directly into the string without the need for string concatenation or string formatting methods like `str.format()`.
"""

"""
my_list = [23,26,27]

my_list.remove(26)
print(my_list)
"""
"""
def add(a):
  
    b = a + 1
    print(a, "if you add one then", b)
    return(b)
add(5)

"""

"""
def type_of_album(artists, album, year_released):

    print(artists, album, year_released)
    if year_released > 1980:
        return "Modern"
    else:
        return "Oldie"
x = type_of_album("Linkon","Artcell",2001)
print(x)
"""
"""

def list_function(list):
    for element in list:
        print(element)

list_function([4,5,6,7,3,7])
"""
"""
# Python Program to Count words in a String using Dictionary
def freq(string):
    
    #step1: A list variable is declared and initialized to an empty list.
    words = []
    
    #step2: Break the string into list of words
    words = string.split() # or string.lower().split()
    
    #step3: Declare a dictionary
    Dict = {}
    
    #step4: Use for loop to iterate words and values to the dictionary
    for key in words:
        Dict[key] = words.count(key)
        
    #step5: Print the dictionary
    print("The Frequency of words is:",Dict)
    
#step6: Call function and pass string in it
freq("Mary had a little lamb Little lamb, little lamb Mary had a little lamb.Its fleece was white as snow And everywhere that Mary went Mary went, Mary went \
Everywhere that Mary went The lamb was sure to go")
"""
"""
# Example for setting param with default value

def isGoodRating(rating=4): 
    if(rating < 7):
        print("this album sucks it's rating is",rating)
        
    else:
        print("this album is good its rating is",rating)

# Test the value with default value and with input

isGoodRating()
isGoodRating(10)
"""
"""
artist = "Michael Jackson"

def printer(artist):
    global internal_var 
    internal_var= "Whitney Houston"
    print(artist,"is an artist")

printer(artist) 
printer(internal_var)

"""
"""
# Example of global variable

myFavouriteBand = "AC/DC"

def getBandRating(bandname):
    if bandname == myFavouriteBand:
        return 10.0
    else:
        return 0.0

print("AC/DC's rating is:", getBandRating("AC/DC"))
print("Deep Purple's rating is:",getBandRating("Deep Purple"))
print("My favourite band is:", myFavouriteBand)
"""
"""

# Example of local variable

def getBandRating(bandname):
    global myFavouriteBand
    myFavouriteBand = "AC/DC"
    if bandname == myFavouriteBand:
        return 10.0
    else:
        return 0.0

print("AC/DC's rating is: ", getBandRating("AC/DC"))
print("Deep Purple's rating is: ", getBandRating("Deep Purple"))
print("My favourite band is", myFavouriteBand)

"""
"""
# Example of global variable and local variable with the same name

myFavouriteBand = "AC/DC"

def getBandRating(bandname):
    myFavouriteBand = "Deep Purple"
    if bandname == myFavouriteBand:
        return 10.0
    else:
        return 0.0

print("AC/DC's rating is:",getBandRating("AC/DC"))
print("Deep Purple's rating is: ",getBandRating("Deep Purple"))
print("My favourite band is:",myFavouriteBand)
"""
"""
def printDictionary(**args):
    for key in args:
        print(key + " : " + args[key])

printDictionary(Country='Canada',Province='Ontario',City='Toronto')
"""
"""

a = 1

try:
    b = int(input("Please enter a number to divide a: "))
    a = a/b
    print("Success a=",a)
except ZeroDivisionError:
    print("The number you provided cant divide 1 because it is 0")
except ValueError:
    print("You did not provide a number")
except:
    print("Something went wrong")
"""
"""
a = 1
b = 0
try:
    #b = int(input("Please enter a number to divide a: "))
    a = a/b
except ZeroDivisionError:
    print("The number you provided cant divide 1 because it is 0")
except ValueError:
    print("You did not provide a number")
except:
    print("Something went wrong")
else:
    print("success a=",a)
finally:
    print("Processing Complete")      
"""
"""
def safe_divide(numerator,denominator):
    try:
        result = numerator / denominator
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero.")
        return None
# Test case
numerator=int(input("Enter the numerator value:-"))
denominator=int(input("Enter the denominator value:-"))
print(safe_divide(numerator,denominator))        
"""
"""
def complex_calculation(num):
    try:
        result = num / (num - 5)
        print (f"Result: {result}")
    except Exception as e:
        print("An error occurred during calculation.")
# Test case
user_input = float(input("Enter a number: "))
complex_calculation(user_input)
"""
"""
list = [2,5,1,3,6,4]
list.reverse()
print(list)
"""

class Car(object):
    # Class attribute (shared by all instances)
    max_speed = 120  # Maximum speed in km/h

    # Constructor method (initialize instance attributes)
    def __init__(self, make, model, color, speed=0):
        self.make = make
        self.model = model
        self.color = color
        self.speed = speed  # Initial speed is set to 0

    # Method for accelerating the car
    def accelerate(self, acceleration):
        if self.speed + acceleration <= Car.max_speed:
            self.speed += acceleration
        else:
            self.speed = Car.max_speed

    # Method to get the current speed of the car
    def get_speed(self):
        return self.speed
    

#Create objects (instances) of the Car class
car1 = Car("Toyota", "Camry", "Blue")
car2 = Car("Honda", "Civic", "Red")
# Accelerate the cars
car1.accelerate(300)
car2.accelerate(20)

# Print the current speeds of the cars
#print(f"{car1.make} {car1.model} is currently at {car1.get_speed()} km/h.")
#print(f"{car2.make} {car2.model} is currently at {car2.get_speed()} km/h.")

print(car2.speed)


