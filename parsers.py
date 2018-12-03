# String parsing libraries
import string

# File parsing libraries
import csv
import json

# File system navigation
from os import listdir

# Database libraries
import sqlite3

################################################################################
# PART #1
################################################################################

    # This function should count the words in an unstructured text document
    # Inputs: A file name (string)
    # Outputs: A dictionary with the counts for each word
    # +1 bonus point for removing punctuation from the wordcounts


def countWordsUnstructured(filename):
    # Initialize a word count dictionary
    wordCounts = {}
    # Open a file and read it
    datafile = open(filename, "r").read()
    # Open the file and read it
    data = datafile.split()

    # Count the words
    for word in data:
        for m in string.punctuation:
            word = word.replace(m, "")
        if word in wordCounts:
            wordCounts[word] = wordCounts[word] + 1
        else:
            wordCounts[word] = 1

    #Return the word count dictionary
    return wordCounts

# Test your part 1 code below.
#bush1989 = countWordsUnstructured("./state-of-the-union-corpus-1989-2017/Bush_1989.txt")
#print (bush1989)


################################################################################
# PART 2
################################################################################
    # This function should transform a dictionary containing word counts to a
    # CSV file. The first row of the CSV should be a header noting:
    # Word, Count
    # Inputs: A word count list and a name for the target file
    # Outputs: A new CSV file named targetfile containing the wordcount data

def generateSimpleCSV(targetfile, wordCounts):

    # Open a file as a csv file
    csvFile = open(targetfile, "w")

    # Create the CSV Writer
    csvWriter = csv.writer(csvFile)

    # Write the header row
    csvWriter.writerow(['Word', 'Count'])

    # Transform the word count dictionary to the content of the csv
    for word in wordCounts:
        csvWriter.writerow([word, wordCounts[word]])

    # Close the file
    csvFile.close()

    # Return a reference to the file
    return csvFile
#

# Test your part 2 code below
#bush1989CSV = generateSimpleCSV('singleFileCSV.csv', bush1989)

# Note that you can test your solution either by using a custom read function or opening the file manually.


################################################################################
# PART 3
################################################################################
    # This function should create a dictionary of word count dictionaries
    # The dictionary should have one dictionary per file in the directory
    # Each entry in the dictionary should be a word count dictionary
    # Inputs: A directory containing a set of text files
    # Outputs: A dictionary containing a word count dictionary for each
    #           text file in the directory

def countWordsMany(directory):

    # Gather a list of file names in the directory
    fileList = listdir(directory)

    # Create an empty dictionary for the word cout dictionaries
    wordCountDict = {}

    # Iterate through the entries and create the dictionary containing the other word count dictionaries for each text file entry
    # Loop through the list of files
    for file in fileList:
        # Count the words in each file
        wordCount = countWordsUnstructured(directory + "/" + file)

        # Add the word counts to the dictionary
        wordCountDict[file] = wordCount

    # Return the dictionary
    return wordCountDict

# Test your part 3 code below
#sotuAddresses = countWordsMany('./state-of-the-union-corpus-1989-2017')


################################################################################
# PART 4
################################################################################
    # This function should create a CSV containing the word counts generated in
    # part 3 with the header:
    # Filename, Word, Count
    # Inputs: A word count dictionary and a name for the target file
    # Outputs: A CSV file named targetfile containing the word count data

def generateDirectoryCSV(wordCounts, targetfile):

    # Open a file as a csv_file
    csvFile = open(targetfile, "w")

    # Create the CSV Writer
    csvWriter = csv.writer(csvFile)

    # Write in the header row as requested
    csvWriter.writerow(['Filename', 'Word', 'Count'])

    # Iterate through the word count directory and add to the csv
    for file in wordCounts:
        for word in wordCounts[file]:
            csvWriter.writerow([file, word, wordCounts[file][word]])

    # Close file
    csvFile.close()

    # Return a pointer to the CSV file
    return csvFile


# Test your part 4 code below
#sotuCSV = generateDirectoryCSV(sotuAddresses, 'directoryCSV.csv')

################################################################################
# PART 5
################################################################################
    # This function should create a JSON containing the word counts generated in
    # part 3. Architect your JSON file such that the hierarchy will allow
    # the user to quickly navigate and compare word counts between files.
    # Inputs: A word count dictionary and a name for the target file
    # Outputs: An JSON file named targetfile containing the word count data

def generateJSONFile(wordCounts, targetfile):

    # Open a file
    jsonFile = open(targetfile, "w")

    # Transform the word count directory to the content of the json
    jsonData = json.dumps(wordCounts)

    # Write out the data
    jsonFile.write(jsonData)

    # Close file
    jsonFile.close()

    #return the json file
    return jsonFile

# Test your part 5 code below
#sotuJSON = generateJSONFile(sotuAddresses, 'directoryJSON.json')

################################################################################
# PART 6
################################################################################
    # This function should search a CSV file from part 4 and find the filename
    # with the largest count of a specified word
    # Inputs: A CSV file to search and a word to search for
    # Outputs: The filename containing the highest count of the target word

def searchCSV(csvFile, word):
    # Create the container variables
    largestFile = ""
    largestCount = 0
    csvDatapointChecked = 0

    # Open the CSV file and create a reader
    csvFile = open(csvFile, "r")
    csvReader = csv.reader(csvFile)

    # Loop through the contents of the CSV
    for line in csvReader:
        # Skip the header
        if line[2] == "Count":
            continue

        # If the line contains the word, compare the word count
        if line[1] == word and int(line[2]) > largestCount:
            largestCount = line[2]
            largestFile = line[0]
        csvDatapointChecked += 1

    # Close the file
    csvFile.close()

    # +1 bonus point for figuring out how many datapoints you had to process to
    # compute this value
    print("The CSV checked " + str(csvDatapointChecked) + " datapoints")

    # Return the filename with the largest wordcounts
    return largestFile


# This function should search a JSON file from part 5 and find the filename
# with the largest count of a specified word
# Inputs: An JSON file to search and a word to search for
# Outputs: The filename containing the highest count of the target word

def searchJSON(JSONfile, word):

    # Tracking variables
    largestFile = ""
    largestCount = 0
    jsonDatapointChecked = 0

    # Open and extract data from the JSON file
    jsonFile = open(JSONfile, "r")
    jsonData = json.load(jsonFile)

    # Check each file for the word
    for datapoint in jsonData:

        # Compare the word count
        if jsonData[datapoint][word] > largestCount:
            largestCount = jsonData[datapoint][word]
            largestFile = datapoint
        jsonDatapointChecked += 1

    # Close the file
    jsonFile.close()


    print("The JSON checked " + str(jsonDatapointChecked) + " datapoints")

    # Return the name of the file with the largest word count
    return largestFile


# Test your part 6 code to find which file has the highest count of a given word
#print("CSV Found: " + searchCSV("directoryCSV.csv", "America"))
#print("JSON Found: " + searchJSON("directoryJSON.json", "America"))

# Friday:
# 2. In the starter code, you also now have additional data on different
#presidents. Design a database schema that allows you to store this data with
#your word counts from the State of the Union addresses. Make sure to consider
#how you would connect between different tables, how to integrate all of the
#data from both sources, and how to minimize the amount of redundant information
#(you should have no duplicate information in your final design).

# One potential solution:
# President
#   int presidentId
#   varchar presidentName
#   int number
#   date startYear
# 	date endYear
#   varchar priorPosition
#   varchar	party
#   varchar vicePresident

# Speeches
#   int presidentId
#   date speechYear
#   varchar word
#   int count

#3. Create a new database using Python as part of your parsers.py code that
# implements this schema. The inputs to this function should be a database name.
def buildDatabase(dbName):
    # Set up a connection to the database
    conn = sqlite3.connect(dbName)
    c = conn.cursor()

    # Ask the connection to execute a SQL statement
    c.execute('''CREATE TABLE president (presidentId int, presidentName varchar,
    presNumber int, startYear date, endYear date, priorPosition varchar,
    party varchar, vicePresident varchar)''')

    c.execute('''CREATE TABLE speeches (presidentId int, speechYear date,
    word varchar, count int)''')

    # Save (commit) the changes
    conn.commit()

    # Close the connection
    conn.close()

#buildDatabase('speechData.db')
